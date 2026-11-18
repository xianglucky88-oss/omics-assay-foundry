from functools import lru_cache
from ..validation import fields, integer, sequence
from ..errors import DomainError

PAIRS = {"AU", "UA", "GC", "CG", "GU", "UG"}

def prepare(payload):
    fields(payload, ["sequence"], ["minimum_loop", "required_pairs", "forbidden_pairs", "beta"])
    seq = payload["sequence"]
    if not isinstance(seq,str) or not 0<=len(seq)<=18 or set(seq)-set("ACGU"):
        raise DomainError("unsupported_sequence", "Use uppercase ACGU sequences of at most 18 bases")
    loop = integer(payload.get("minimum_loop",3), "minimum_loop", minimum=0, maximum=18)
    def pairs(name):
        result = set()
        for pair in sequence(payload.get(name,[]), name, maximum=153):
            pair = sequence(pair,name,minimum=2,maximum=2)
            i,j = (integer(x,name,minimum=0,maximum=max(0,len(seq)-1)) for x in pair)
            if i >= j or j>=len(seq):
                raise DomainError("invalid_pair", "Pairs must satisfy 0 <= i < j < sequence length")
            result.add((i,j))
        return result
    required, forbidden = pairs("required_pairs"), pairs("forbidden_pairs")
    if required & forbidden:
        raise DomainError("constraint_conflict", "A pair cannot be both required and forbidden")
    occupied = [x for pair in required for x in pair]
    if len(occupied)!=len(set(occupied)):
        raise DomainError("constraint_conflict", "A base cannot have two required partners")
    if any(i<k<j<l or k<i<l<j for i,j in required for k,l in required):
        raise DomainError("unsupported_pseudoknot", "Crossing required pairs are outside the noncrossing model")
    if any(seq[i]+seq[j] not in PAIRS or j-i-1<loop for i,j in required):
        raise DomainError("constraint_conflict", "A required pair is incompatible with pairing or loop rules")
    return seq, loop, required, forbidden

def enumerate_structures(seq, minimum_loop, required, forbidden):
    generated = [0]
    @lru_cache(None)
    def generate(left,right):
        if left>=right:
            return ((),)
        values = list(generate(left+1,right))
        for partner in range(left+minimum_loop+1,right):
            if seq[left]+seq[partner] not in PAIRS or (left,partner) in forbidden:
                continue
            for inside in generate(left+1,partner):
                for after in generate(partner+1,right):
                    generated[0] += 1
                    if generated[0] > 200000:
                        raise DomainError("ensemble_limit", "Exact structure enumeration exceeds the supported work budget")
                    values.append(((left,partner),)+inside+after)
        return tuple(values)
    return [structure for structure in generate(0,len(seq)) if required.issubset(structure)]
