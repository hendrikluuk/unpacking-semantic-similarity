"""
  Given a list of {m} lists and total sample size {n}, from each sublist {i} sample {n_i} elements such that {n_i} would be as close as possible to {n / m}.
  You need to take into account that some sublists may be much shorter than {n / m}.
"""
import random
from copy import deepcopy

def fair_coin(p:float) -> bool:
    """
    Returns:
        bool: True for heads, False for tails.
    """
    return random.random() < p

def uniform_sample(lists: list[list[dict]], n: int, verbose:bool=False) -> list[dict]:
    """
        Sample as uniformly as possible from multiple lists.
        Return a list of {n} examples.
    """
    if len(lists) == 0 or n <= 0:
        return []

    # Avoid modifying the original lists
    lists = [(i, l) for i, l in enumerate(lists)]

    result = []
    exhausted_lists = []
    n_per_list = [0] * len(lists)
    while len(result) < n:
        for i, uid_list_tuple in enumerate(lists):
            unique_index, l = uid_list_tuple
            random.shuffle(l)

            distance_from_goal = n - len(result)
            if distance_from_goal <= 0:
                break

            # number of lists left
            m = len(lists) - i
            # number of elements correponding to uniform sampling from all lists
            ideal_n = distance_from_goal / m

            # number of elements we can actually sample from this sublist
            sublist_goal = min(len(l), ideal_n)
            if sublist_goal == len(l):
                exhausted_lists.append(uid_list_tuple)
            # sample additional element with corresponding probability if the ideal_n is not an integer
            sublist_goal = min(len(l), int(sublist_goal) + fair_coin(sublist_goal - int(sublist_goal)))

            result.extend(l[:sublist_goal])
            n_per_list[unique_index] += sublist_goal
            # Remove sampled elements from the sublist
            del l[:sublist_goal]

        for l in exhausted_lists:
            lists.remove(l)
        
        if len(lists) == 0:
            break
        exhausted_lists = []

    if verbose:
        print(f"Sampled {len(result)} examples from {len(n_per_list)} lists.")
        print(f"Number of elements sampled from each list: {n_per_list}") 
    return result[:n]