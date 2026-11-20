from ..validation import fields, mapping, sequence, label, number
from ..errors import DomainError

MAX_LEAVES = 32
MAX_SITES = 2000
MAX_DEPTH = 32
MAX_NODES = 127
MAX_BRANCH = 1000

def parse_tree(root):
    """Validate and copy a rooted candidate tree.

    A node with ``name`` is a leaf; otherwise it has at least two children.
    The branch length stored on the root is retained for symmetry with the
    original input format but is never used by the likelihood calculation.
    """
    leaves = []
    nodes = [0]

    def visit(node, depth):
        if depth > MAX_DEPTH:
            raise DomainError("tree_depth", "Tree depth exceeds 32")
        nodes[0] += 1
        if nodes[0] > MAX_NODES:
            raise DomainError("tree_size", "Tree has too many nodes")
        mapping(node, "tree node")
        if "name" in node:
            fields(node, ["name"], ["length"])
            name = label(node["name"], "leaf")
            leaves.append(name)
            return {"name": name,
                    "length": number(node.get("length", 0), "branch length", minimum=0, maximum=MAX_BRANCH)}
        fields(node, ["children"], ["length"])
        children = sequence(node["children"], "children", minimum=2, maximum=MAX_LEAVES)
        return {"children": [visit(x, depth + 1) for x in children],
                "length": number(node.get("length", 0), "branch length", minimum=0, maximum=MAX_BRANCH)}

    return visit(root, 0), leaves

def prepare(payload):
    fields(payload, ["tree", "sequences"])
    seqs = mapping(payload["sequences"], "sequences")
    if not 2 <= len(seqs) <= MAX_LEAVES:
        raise DomainError("size_limit", "Use two to 32 leaves")
    lengths = set()
    for name, seq in seqs.items():
        label(name, "leaf")
        if not isinstance(seq, str) or not 1 <= len(seq) <= MAX_SITES or set(seq) - set("ACGT"):
            raise DomainError("unsupported_sequence", "Use complete uppercase ACGT sequences of length 1 to 2000")
        lengths.add(len(seq))
    if len(lengths) != 1:
        raise DomainError("unaligned_sequences", "All sequences must have the same length")
    tree, leaves = parse_tree(payload["tree"])
    if len(leaves) != len(set(leaves)) or set(leaves) != set(seqs):
        raise DomainError("leaf_mismatch", "Tree leaves must match sequences exactly once")
    return tree, seqs
