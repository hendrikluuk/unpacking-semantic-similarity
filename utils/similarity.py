from jarowinkler import jarowinkler_similarity as jarowinkler
from .token_similarity import token_similarity

def ss_asym(a:list|set, b:list|set) -> float:
    """
    Calculate the asymmetric semantic similarity between concepts {a} and {b}.

    If {b} is a subset of {a} then {ss_asym(a,b)} returns 1.0 indicating that 
    {a} implies {b} completely (100%).

    If {a} is a subset of {b} then {ss_asym(a,b)} returns a value that reflects
    the fraction of the referents of {b} that are implicated by {a}. For example,
    if {a} has a single referent that overlaps with the 10 referents of {b}, then
    {ss_asym(a,b) = 0.1}. This can be interpreted as: {a} implies {b} to the extent
    of 10%.
    """
    if isinstance(a, list):
        a = set(a)
    if isinstance(b, list):
        b = set(b)
    return len(b.intersection(a)) / len(b)

def ss_sym(a:list|set, b:list|set) -> float:
    """
    Calculate the symmetric semantic similarity between concepts {a} and {b}.
    """
    if isinstance(a, list):
        a = set(a)
    if isinstance(b, list):
        b = set(b)
    return len(a.intersection(b)) / len(a.union(b))