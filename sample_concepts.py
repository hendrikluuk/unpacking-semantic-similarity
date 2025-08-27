#!/usr/bin/env python3
"""
Sample equivalent and parent-child concepts and relations from the corresponding annotation files.
"""
import json
import argparse
import itertools
from typing import Iterable
from random import sample, randint

from utils.loaders import load_equivalent_concepts, load_related_concepts, load_relational_domains
from utils.similarity import jarowinkler, token_similarity
from utils.uniform_sampler import uniform_sample

concepts = {}
record_template = {
    # symbol for concept {a}
    "a": "",
    # symbol for concept {b}
    "b": "",
    # unidirectional semantic entailment (a entails b)
    "a -> b": 1.0,
    # unidirectional semantic entailment (b entails a)
    "b -> a": 1.0,
    # symbolic similarity on character level
    "char(a ~ b)": 0.0,
    # symbolic similarity on token level
    "token(a ~ b)": 0.0,
    # domain of knowledge
    "domain": "",
}

def init():
    """
    Sample equivalent and parent-child concepts and relations from the corresponding annotation files.
    Create separate subsets of examples for different aspects of semantic similarity.
    """
    concepts["equivalent"] = load_equivalent_concepts()
    concepts["parent-child"] = load_related_concepts()
    concepts["relations"] = load_relational_domains()

def sample_pair(pool:Iterable) -> list[dict]:
    """
    Sample concepts from the loaded datasets.
    """
    assert len(pool) > 1, "Pool must contain at least two elements to sample a pair."
    # generate random integer from 0 to len(pool) - 1
    a = randint(0, len(pool) - 1) 
    b = randint(0, len(pool) - 1)

    while a == b:
        b = randint(0, len(pool) - 1)

    return (pool[a], pool[b]) 

def symbolic_similarity(a:str, b:str, target:dict):
    target["char(a ~ b)"] = jarowinkler(a, b)
    target["token(a ~ b)"] = token_similarity(a, b)

def sample_equivalent_concepts(n_max:int = 100) -> list[dict]:
    result = []
    pool = concepts["equivalent"]
    for record in sample(pool, min(n_max, len(pool))):
        example = record_template.copy()
        example["a"] = record["label1"]
        example["b"] = record["label2"]
        example["domain"] = record["domain"]
        symbolic_similarity(record["label1"], record["label2"], example)
        result.append(example)
    return result

def sample_parent_child_concepts(n_max:int = 100) -> list[dict]:
    result = []
    pool = concepts["parent-child"]
    for record in sample(pool, min(n_max, len(pool))):
        example = record_template.copy()
        example["a"] = record["parent"]["label"]
        example["b"] = record["child"]["label"]
        example["a -> b"] = record["ss_asym_parent_child"]
        example["b -> a"] = record["ss_asym_child_parent"]
        symbolic_similarity(record["parent"]["label"], record["child"]["label"], example)
        example["domain"] = record["domain"]
        result.append(example)
    return result

def sample_unrelated_concepts(n_max:int = 100) -> list[dict]:
    """
    Sample [n_max] pairs of unrelated concepts.
    """
    result = []
    pool = concepts["equivalent"]
    for i in range(min(n_max, len(pool))):
        a, b = sample_pair(pool)
        example = record_template.copy()
        example["a"] = a["label1"]
        example["b"] = b["label1"]
        # assuming that randomly sampled pairs are unrelated
        example["a -> b"] = 0.0
        example["b -> a"] = 0.0
        symbolic_similarity(a["label1"], b["label1"], example)
        example["domain"] = f'{a["domain"]}/{b["domain"]}'
        result.append(example)
    return result

def sample_equivalent_relations(n_max:int = 100) -> list[dict]:
    """
    Sample [n_max] pairs of equivalent relations.
    """
    result = []
    pool = concepts["relations"]

    for relational_domain in pool:
        for subset in relational_domain:
            for a,b in itertools.combinations(subset["predicates"], 2):
                example = record_template.copy()
                example["a"] = a
                example["b"] = b
                example["domain"] = subset["title"]
                symbolic_similarity(a, b, example)
                result.append(example)

    return sample(result, min(n_max, len(result)))

def sample_contradictory_relations(n_max:int = 100) -> list[dict]:
    """
    Sample [n_max] pairs of contradictory relations.
    """
    result = []
    pool = concepts["relations"]

    for relational_domain in pool:
        # get all combinations of contradictory predicate pairs from the domain 
        for ntuple in itertools.product(*list(map(lambda x: zip(x["predicates"], [x["title"]]*len(x["predicates"])), relational_domain))):
            if len(ntuple) != 2:
                continue
            for a, b in itertools.combinations(ntuple, 2):
                example = record_template.copy()
                example["a"] = a[0]
                example["b"] = b[0]
                example["domain"] = f"{a[1]}/{b[1]}"
                example["a -> b"] = -1.0
                example["b -> a"] = -1.0
                symbolic_similarity(a[0], b[0], example)
                result.append(example)

    return sample(result, min(n_max, len(result)))

def sample_unrelated_relations(n_max:int = 100) -> list[dict]:
    result = []
    pool = concepts["relations"]

    for domain_a, domain_b in itertools.combinations(pool, 2):
        # two lists of unrelated predicates
        ab_predicates = []
        for domain in [domain_a, domain_b]:
            ab_predicates.append([(predicate, subset["title"]) for subset in domain for predicate in subset["predicates"]])

        # get all combinations of unrelated predicates
        for a, b in itertools.product(*ab_predicates):
            example = record_template.copy()
            example["a"] = a[0]
            example["b"] = b[0]
            example["domain"] = f"{a[1]}/{b[1]}"
            example["a -> b"] = 0.0
            example["b -> a"] = 0.0
            symbolic_similarity(a[0], b[0], example)
            result.append(example)

    return sample(result, min(n_max, len(result)))

sampler = {
    "equivalent concepts": sample_equivalent_concepts,
    "parent-child concepts": sample_parent_child_concepts,
    "unrelated concepts": sample_unrelated_concepts,
    "equivalent relations": sample_equivalent_relations,
    "contradictory relations": sample_contradictory_relations,
    "unrelated relations": sample_unrelated_relations,
}

def sample_concepts(n_max:int) -> list[dict]:
    result = []
    for subset in sampler:
        result.append(sampler[subset](n_max))
    result = uniform_sample(result, n_max, verbose=True)
    print(f"Sampled {len(result)} concepts from {len(sampler)} subsets: {', '.join(sampler.keys())}")
    return result

def main(n_max:int):
    """
    Main function to run the sampling of concepts.
    """
    init()

    sampled_concepts = sample_concepts(n_max)
    outfile = f"out/sampled_concepts_{n_max}.json"
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(sampled_concepts, f, indent=4)
    print(f"Saved the sample to '{outfile}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample concepts and relations from annotated datasets.")
    parser.add_argument("--n_max", type=int, default=2500, help="Maximum number of samples to draw from each subset.")
    args = parser.parse_args()
    main(args.n_max)
