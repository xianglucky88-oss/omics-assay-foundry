"""Input contract and graph validation for the joint pedigree model.

A pedigree is a directed acyclic graph: edges run from parent to child.
A member may be reached from one ancestor through more than one path; that is
an inbreeding (common-ancestor) loop, not an illegal parent cycle, and is
supported. Genuine directed cycles are rejected with ``illegal_cycle``.
"""
from ..validation import fields, label, mapping, number, sequence
from ..errors import DomainError

DIPLOID = 2
HAPLOID = 1
MAX_MEMBERS = 64


def _component_of(index, members):
    """Undirected connected component of the parent/child graph."""
    index_of = {member["id"]: i for i, member in enumerate(members)}
    adjacency = [set() for _ in members]
    for i, member in enumerate(members):
        for parent in member["parents"]:
            j = index_of[parent]
            adjacency[i].add(j)
            adjacency[j].add(i)
    seen = set()
    stack = [index]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(adjacency[current] - seen)
    return seen


def _topological_order(members):
    """Kahn ordering; raises ``illegal_cycle`` when parent edges form a cycle."""
    ids = [member["id"] for member in members]
    index = {name: i for i, name in enumerate(ids)}
    indegree = [0] * len(members)
    children = [[] for _ in members]
    for i, member in enumerate(members):
        for parent in member["parents"]:
            indegree[i] += 1
            children[index[parent]].append(i)
    ready = [i for i, degree in enumerate(indegree) if degree == 0]
    ordered = []
    while ready:
        current = ready.pop()
        ordered.append(current)
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(ordered) != len(members):
        raise DomainError("illegal_cycle", "Parent references form a directed cycle; a pedigree must be acyclic")
    return ordered


def _common_ancestors(members):
    """Structural inbreeding common ancestors.

    A member is a common ancestor when some descendant can be reached from it
    through two distinct child subtrees (the two children's descendant sets
    intersect). Reaching a member twice this way is an inbreeding loop, not a
    directed cycle, so it is supported rather than rejected.
    """
    descendants = [None] * len(members)

    def reachable(i):
        if descendants[i] is not None:
            return descendants[i]
        result = {i}
        for child in (k for k, member in enumerate(members) if members[i]["id"] in member["parents"]):
            result |= reachable(child)
        descendants[i] = result
        return result

    common = []
    for i, member in enumerate(members):
        child_indices = [k for k, other in enumerate(members) if members[i]["id"] in other["parents"]]
        shared = any(reachable(a) & reachable(b)
                     for position, a in enumerate(child_indices)
                     for b in child_indices[position + 1:])
        if shared:
            common.append(member["id"])
    return common


def prepare(payload):
    fields(payload, ["allele_frequency", "members"], ["max_states"])
    frequency = number(payload["allele_frequency"], "allele_frequency", minimum=0, maximum=1)
    max_states = number(payload.get("max_states", 500_000), "max_states", minimum=1, maximum=5_000_000)
    rows = sequence(payload["members"], "members", minimum=1, maximum=MAX_MEMBERS)

    members = []
    seen_ids = set()
    for row in rows:
        mapping(row, "member")
        fields(row, ["id"], ["ploidy", "parents", "source_parent", "likelihoods"])
        name = label(row["id"], "member id")
        if name in seen_ids:
            raise DomainError("duplicate_member", f"Member id {name} is declared more than once")
        seen_ids.add(name)

        ploidy = row.get("ploidy", DIPLOID)
        if ploidy not in (HAPLOID, DIPLOID):
            raise DomainError("invalid_ploidy", f"Member {name} must declare ploidy 1 or 2")

        parents = row.get("parents", [])
        sequence(parents, f"parents of {name}", minimum=0, maximum=2)
        if any(not isinstance(parent, str) for parent in parents):
            raise DomainError("invalid_parent", f"Member {name} parents must be member id strings")
        if len(set(parents)) != len(parents):
            raise DomainError("invalid_parent", f"Member {name} lists the same parent twice")
        if ploidy == DIPLOID and len(parents) not in (0, 2):
            raise DomainError("invalid_parent", f"Diploid member {name} needs zero or two parents")
        if ploidy == HAPLOID and len(parents) not in (1, 2):
            raise DomainError("invalid_parent", f"Haploid member {name} needs one parent or two with a source parent")

        source_parent = row.get("source_parent")
        if source_parent is not None:
            if not isinstance(source_parent, str):
                raise DomainError("invalid_source_parent", f"Member {name} source_parent must be a member id string")
            if source_parent not in parents:
                raise DomainError("invalid_source_parent", f"Member {name} source_parent must be one of its parents")
        if len(parents) == 1 and ploidy == HAPLOID:
            source_parent = parents[0]
        if ploidy == HAPLOID and len(parents) == 2 and source_parent is None:
            raise DomainError("invalid_source_parent", f"Haploid member {name} with two parents needs source_parent")

        likelihoods = None
        if "likelihoods" in row:
            values = sequence(row["likelihoods"], f"likelihoods of {name}", minimum=ploidy + 1, maximum=ploidy + 1)
            likelihoods = tuple(number(value, f"likelihoods of {name}", minimum=0) for value in values)

        members.append({"id": name, "ploidy": ploidy, "parents": list(parents),
                        "source_parent": source_parent, "likelihoods": likelihoods,
                        "domain": ploidy + 1})

    for member in members:
        unknown = set(member["parents"]) - seen_ids
        if unknown:
            raise DomainError("unknown_parent", f"Member {member['id']} references unknown parent: {', '.join(sorted(unknown))}")

    order = _topological_order(members)
    observed = [member["id"] for member in members if member["likelihoods"] is not None]

    components = []
    assigned = set()
    for i in range(len(members)):
        if i in assigned:
            continue
        group = _component_of(i, members)
        assigned |= group
        components.append(group)

    index_of = {member["id"]: i for i, member in enumerate(members)}
    for member in members:
        own = next(group for group in components if index_of[member["id"]] in group)
        group_observed = any(members[k]["likelihoods"] is not None for k in own)
        if member["likelihoods"] is not None:
            member["support"] = "observed"
        elif group_observed:
            member["support"] = "inferred"
        else:
            member["support"] = "prior_only"

    return {"frequency": frequency, "max_states": int(max_states), "members": members,
            "order": order, "observed": observed,
            "common_ancestors": _common_ancestors(members)}
