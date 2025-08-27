#!/usr/bin/env python3
"""
  Load propositions from annotations/annotated_propositions.json

    1. bidirectional semantic entailment (a <-> b)
        1.1 equivalent but distinct subjects
        1.2 equivalent but distinct objects
        1.3 equivalent but distinct predicates
        1.4 equivalent but distinct subjects and objects
        1.5 equivalent but distinct subjects and predicates
        1.6 equivalent but distinct objects and predicates
        1.7 equivalent but distinct subjects, objects and predicates

    2. unidirectional semantic entailment (a -> b)
        2.1 one-way implication between subjects, identical objects and predicates
        2.2 one-way implication between objects, identical subjects and predicates
        # one-way implication between subjects, identical objects and predicates
        2.3 one-way implication between predicates, identical subjects and objects

    3. no semantic entailment (a || b)
        3.1 unrelated subjects, identical objects and predicates
        3.2 unrelated objects, identical subjects and predicates
        3.3 unrelated predicates, identical subjects and objects  
    
    Create separate datasets for 1, 2, and 3 above by random sampling.
    In each dataset, strive for a balanced distribution of the different categories.
"""
import json
import math
import argparse

from utils.loaders import load_propositions
from utils.uniform_sampler import uniform_sample
from utils.similarity import jarowinkler, token_similarity

propositions = None

record_template = {
    # proposition {a}
    "a": "",
    # proposition {b}
    "b": "",
    # unidirectional semantic entailment (a entails b)
    "a -> b": 1.0,
    # unidirectional semantic entailment (b entails a)
    "b -> a": 1.0,
    # symbolic similarity on character level
    "char(a ~ b)": 0.0,
    # symbolic similarity on token level
    "token(a ~ b)": 0.0,
    # class of proposition
    "domain": "",
}

def init():
    global propositions
    propositions = load_propositions("annotations/annotated_propositions.json")

def sample_propositions(n_max:int=1000) -> list[dict]:
    result = []
    for prop_class in propositions:
        class_template = record_template.copy()
        if prop_class[0] == '2':
            class_template["b -> a"] = 0.0
        elif prop_class[0] == '3':
            class_template["a -> b"] = class_template["b -> a"] = -1.0
        elif prop_class[0] == '4':
            class_template["a -> b"] = class_template["b -> a"] = 0.0
        class_template["domain"] = prop_class

        class_sample = []
        for prop in propositions[prop_class]:
            example = class_template.copy()
            example["a"] = prop["A"]
            example["b"] = prop["B"]
            example["char(a ~ b)"] = jarowinkler(prop["A"], prop["B"])
            example["token(a ~ b)"] = token_similarity(prop["A"], prop["B"])
            # calculate semantic entailment scores
            for key in ["a -> b", "b -> a"]:
                example[key] = math.prod([prop[sop][key] for sop in ["subject", "object", "predicate"]])
            class_sample.append(example)
        
        result.append(class_sample)
    
    result = uniform_sample(result, n_max, verbose=True)
    print(f"Sampled {len(result)} propositions from the following classes: {', '.join(propositions.keys())}")
    return result

def main(n_max:int=1500):
    init()
    result = sample_propositions(n_max)
    outfile = f"out/sampled_propositions_{n_max}.json"
    with open(outfile, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Sampled {len(result)} propositions and saved to '{outfile}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample propositions from annotated propositions.")
    parser.add_argument("--n_max", type=int, default=1500, help="Maximum number of samples to return.")
    args = parser.parse_args()
    main(args.n_max)