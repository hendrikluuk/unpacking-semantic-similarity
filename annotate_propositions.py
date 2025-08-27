#!/usr/bin/env python3
"""
Annotate the semantic and symbolic (character-level) similarity of propositions on subject, object, predicate and proposition levels.
"""
import re
import json
import traceback
import itertools

from utils.loaders import load_equivalent_concepts, load_related_concepts, load_relational_domains, load_propositions, load_proposition_constraints
from utils.similarity import ss_asym, jarowinkler, token_similarity

concept_index = {}
relational_similarity_index = {}

not_brackets = r"[^\[\]]+"
sop_regex = re.compile(rf"(\[{not_brackets}\]){not_brackets}(\[{not_brackets}\]){not_brackets}(\[{not_brackets}\]).*")

def extract_spo(proposition: str) -> list[str]:
    """
    Extract subject, object, and predicate from a proposition string.
    Returns a tuple of (subject, object, predicate).
    """
    matches = sop_regex.findall(proposition)
    try:
        if matches:
            return [re.sub(r"\[|\]", "", r) for r in matches[0]]
        else:
            raise ValueError(f"Proposition does not match expected format: '{proposition}'")
    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Proposition extraction failed from: '{proposition}'") from e

def concept_similarity(a: str, b: str) -> dict:
    """
    Calculate the similarity between two concepts based on their labels {a} and {b}.
    If the concepts are equivalent, return 1.0, otherwise return 0.0.
    """
    ss_ab = 0.0
    ss_ba = 0.0

    parent_child_key = f"{a}#{b}"
    # equivalent concepts have domain as the value in the index
    domain = concept_index.get((a, b)) or concept_index.get((b, a))

    if a == b:
        # identical labels imply identical concepts
        ss_ab = ss_ba = 1.0
    elif domain:
        # we are dealing with distinct labels of the same concept
        ss_ab = ss_ba = 1.0
    elif concept_index.get(parent_child_key, {}):
        # we are dealing with nested concepts
        a_referents = concept_index[parent_child_key]["parent"]
        b_referents = concept_index[parent_child_key]["child"]
        domain = concept_index[parent_child_key]["domain"]
        ss_ab = ss_asym(a_referents, b_referents)
        ss_ba = ss_asym(b_referents, a_referents)
    else:
        domain = f"{concept_index[a]}/{concept_index[b]}"

    return {
        "a": a,
        "b": b,
        "a -> b": ss_ab,
        "b -> a": ss_ba,
        "char(a ~ b)": jarowinkler(a, b), 
        "token(a ~ b)": token_similarity(a, b),
        "domain": domain
    }

def relational_similarity(a: str, b: str) -> dict:
    """
    Return the relational similarity between two predicates.
    """
    score = 0.0
    if a == b:
        # identical predicates
        score = 1.0
    score, domain = relational_similarity_index.get((a, b)) or relational_similarity_index.get((b, a), (-100, "unknown"))

    return {
        "a": a,
        "b": b,
        "a -> b": score, 
        "b -> a": score, 
        "char(a ~ b)": jarowinkler(a, b), 
        "token(a ~ b)": token_similarity(a, b),
        "domain": domain
    }

def normalize(x:list[str]) -> set[str]|str:
    if isinstance(x, str):
        # we need to replace brackets to simplify SOP parsing
        return x.lower().replace("[", "(").replace("]", ")").strip()
    return set(map(lambda x: x.lower(), x))

def init_concept_index():
    global concept_index

    equivalent_concepts = load_equivalent_concepts()
    for concept in equivalent_concepts:
        key = (normalize(concept["label1"]), normalize(concept["label2"]))
        concept_index[key] = concept_index[tuple(reversed(key))] = concept["domain"]
        concept_index[(key[0], key[0])] = concept_index[(key[1], key[1])] = concept["domain"]
        concept_index[key[0]] = concept_index[key[1]] = concept["domain"]

    parent_child_concepts = load_related_concepts()
    for concept in parent_child_concepts:
        parent_child_key = f"{normalize(concept['parent']['label'])}#{normalize(concept['child']['label'])}"
        concept_index[parent_child_key] = {
            "parent": normalize(concept["parent"]["referents"]),
            "child": normalize(concept["child"]["referents"]),
            "domain": concept["domain"]
        }

def init_relational_similarity():
    rd = load_relational_domains()
    for domain in rd:
        contradictory_subsets = itertools.combinations(domain, 2)
        # contradictory predicates within the same domain
        for a,b in contradictory_subsets:
            for pair in itertools.product(a["predicates"], b["predicates"]):
                value = (-1.0, f"{a['title']}/{b['title']}")
                relational_similarity_index[pair] = relational_similarity_index[tuple(reversed(pair))] = value 

        # equivalent predicates within the same domain
        for subset in domain:
            value = (1.0, subset["title"])
            for a, b in itertools.product(subset["predicates"], repeat=2):
                relational_similarity_index[(a, b)] = value

    for domain_a, domain_b in itertools.combinations(rd, 2):
        # unrelated predicates between different domains
        ab_predicates = []
        for domain in [domain_a, domain_b]:
            ab_predicates.append([(predicate, subset["title"]) for subset in domain for predicate in subset["predicates"]])

        for a, b in itertools.product(*ab_predicates):
            relational_similarity_index[(a[0], b[0])] = (0.0, f"{a[1]}/{b[1]}")
            relational_similarity_index[(b[0], a[0])] = (0.0, f"{b[1]}/{a[1]}")

def is_equivalent(pair:dict) -> bool:
    return pair["a -> b"] == pair["b -> a"] == 1.0    

def is_unrelated(pair:dict) -> bool:
    return pair["a -> b"] == pair["b -> a"] == 0    

def is_contradictory(pair:dict) -> bool:
    return pair["a -> b"] == pair["b -> a"] == -1.0    

def is_nested(pair:dict) -> bool:
    return pair["a -> b"] < 1.0 and pair["b -> a"] == 1.0

def is_same_symbol(pair:dict) -> bool:
    return pair["a"].lower() == pair["b"].lower()

def validate_helper(pair:dict, constraints:dict[str]) -> dict:
    """
    Validate a proposition pair against the given constraints.

    Constraints are specified as:
    {
        "object/subject": "S" (synonymous), "I" (identical), "N" (nested), "U" (unrelated),
        "predicate": "S" (synonymous), "I" (identical), "C" (contradictory), "U" (unrelated)
    }
    """
    checks = {}
    try:
        for key in ["subject", "object", "predicate"]:
            if constraints[key] == "I":
                # identical concepts
                checks[key] = is_equivalent(pair[key]) and is_same_symbol(pair[key])
            elif constraints[key] == "S":
                # synonymous concepts with distinct labels
                checks[key] = is_equivalent(pair[key]) and not is_same_symbol(pair[key]) 
            elif constraints[key] == "N":
                # nested concepts ({b} is a member of {a})
                checks[key] = is_nested(pair[key]) and not is_same_symbol(pair[key])
            elif constraints[key] == "C":
                # contradictory keys
                checks[key] = is_contradictory(pair[key]) and not is_same_symbol(pair[key])
            elif constraints[key] == "U":
                # unrelated concepts
                checks[key] = is_unrelated(pair[key]) and not is_same_symbol(pair[key]) 
    except Exception as e:
        traceback.print_exc()
        print(f"Validation failed for proposition pair: {pair}")
        print(f"Constraints: {constraints}")
        raise ValueError(f"Validation error: {e}") from e
    return checks

def validate_proposition_pair(pair:dict, constraints:dict[str]) -> bool:
    """
    Make sure that the semantic and symbolic similarities between the two propositions meet
    the constraints of the corresponding proposition class.
    """
    checks = validate_helper(pair, constraints)
    status = all(checks.values())
    if not status:
        print(f"Validation failed for proposition pair: {pair}")
        print(f"Constraints: {constraints}")
        print(f"Checks: {checks}")
    return status

def init():
    """
    Initialize the concept index and relational similarity index.
    """
    init_concept_index()
    init_relational_similarity()
    print(f"Initialized concept index with {len(concept_index)} concepts.")
    print(f"Initialized relational similarity index with {len(relational_similarity_index)} relations.")

def main():
    """
    Main function to initialize the indices.
    """
    init()
    propositions = load_propositions()
    constraints = load_proposition_constraints(format="for validation")
    print(f"Loaded {len(propositions)} classes of propositions.")

    counter = 0
    try:
        for prop_class in propositions:
            print(f"Processing propositions of class '{prop_class}'...")
            for proposition_pair in propositions[prop_class]:
                spo_a = extract_spo(proposition_pair["A"])
                spo_b = extract_spo(proposition_pair["B"])

                argument_order = proposition_pair.get("argument_order", None)
                if argument_order == "sop":
                    # swap predicate and object to conform to spo order
                    spo_a = (spo_a[0], spo_a[2], spo_a[1]) 
                    spo_b = (spo_b[0], spo_b[2], spo_b[1]) 

                update = {}
                for i, key in enumerate(["subject", "predicate", "object"]):
                    # Calculate similarities
                    if key == "predicate":
                        update[key] = relational_similarity(normalize(spo_a[i]), normalize(spo_b[i]))
                    else:
                        update[key] = concept_similarity(normalize(spo_a[i]), normalize(spo_b[i]))

                proposition_pair.update(update)
                status = validate_proposition_pair(proposition_pair, constraints[proposition_pair["type"]])
                proposition_pair["valid"] = status
                counter += 1
                assert status, f"Proposition {counter} validation failed for {proposition_pair}"
    except KeyboardInterrupt:
        print("Interrupted by user. Saving current state...")

    outfile = "annotations/annotated_propositions.json" 
    with open(outfile, "w") as f:
        json.dump(propositions, f, indent=4)
    print(f"Validated {counter} propositions. Saved results to '{outfile}'.")

if __name__ == "__main__":
    main()