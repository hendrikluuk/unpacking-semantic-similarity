#!/usr/bin/env python3
"""
Generate semantically related proposition pairs based on annnotations in 'annotations' folder.

Gpt4.1 was chosen as the model for proposition generation, because gpt-4.1-mini generated very primitive sentences and
claude-3-5-sonnet sometimes included both labels in the example.
"""
import json
import argparse
import traceback
from random import sample

from utils.loaders import load_equivalent_concepts, load_relational_domains, load_related_concepts, load_proposition_constraints
from utils import call_llm

call = "generate-propositions"
model = "gpt-4.1"

def sample_concepts(concepts: list[dict|list[dict]], exclude: list[dict]=[], k:int=3) -> list[dict]:
    """
    Sample n distinct concepts making sure all domains are represented equally."""
    assert k % 3 == 0, "n must be a multiple of 3 to sample equally from three domains."
    sampled_concepts = []

    if len(concepts):
        if all(isinstance(c, dict) for c in concepts):
            domains = ["chemistry", "biology", "medicine"]
            for domain in domains:
                candidates = sample(list(filter(lambda x: x['domain'] == domain and x not in exclude, concepts)), int(k/len(domains)))
                sampled_concepts.extend(candidates)
        elif all(isinstance(c, list) for c in concepts):
            # relational domains have no subject domain annotation
            sampled_concepts = sample(concepts, k)

    return sampled_concepts

def prune(concepts: list[dict]) -> list[dict]:
    """ Remove unnecessary attributes from concepts """
    assert isinstance(concepts, list) and all(isinstance(c, dict) for c in concepts), "Concepts must be a list of dictionaries."
    include = ["parent", "child", "label", "domain", "label1", "label2"]

    result = []
    for concept in concepts:
        concept = {k: v for k, v in concept.items() if k in include}
        for key in ["parent", "child"]:
            if key in concept and isinstance(concept[key], dict):
                concept[key] = {k: v for k, v in concept[key].items() if k in include}
        result.append(concept)

    print(f"Pruned concepts: {result[:5]}... Total: {len(result)}")
    return result 

def generate_propositions(n:int=3, k:int=3) -> list[dict]:
    """
    Iterate through tasks.
    For each task generate 100 propositions based on the following scheme:
      Sample k distinct concepts for the subject and k distinct ones for the object (each from a different subject domain).
      Sample k distinct predicates from the relational domains.
      Generate context object based on the task and sampled concepts and predicates.
      Call llm with the context object to generate propositions.
      If the returned object is not empty, append it to the propositions list.
    Return the list of propositions.
    """
    equivalent_concepts = prune(load_equivalent_concepts())
    related_concepts = prune(load_related_concepts())
    relational_domains = load_relational_domains()
    tasks = load_proposition_constraints()
    
    propositions = {}

    try:
        for major_key in tasks:
            for minor_key in tasks[major_key]:
                if minor_key == "label":
                    continue

                task = f"{major_key}. {tasks[major_key]['label']}\n  {minor_key}. {tasks[major_key][minor_key]['label']}\n" +\
                        f"   A. {tasks[major_key][minor_key]['A']}" +\
                        f"   B. {tasks[major_key][minor_key]['B']}"

                subject_pool = object_pool = equivalent_concepts
                predicate_pool = relational_domains
                # retrieve a predicate hierarchy (out of scope for now)
                predicate_hierarchy = []

                if minor_key == "2.1":
                    subject_pool = related_concepts
                elif minor_key == "2.2":
                    object_pool = related_concepts
                elif minor_key == "2.3":
                    predicate_pool = predicate_hierarchy
                    continue

                if not minor_key in propositions:
                    propositions[minor_key] = []

                while len(propositions[minor_key]) < n:
                    subjects = sample_concepts(subject_pool, k=k)
                    objects = sample_concepts(object_pool, subjects, k=k)
                    predicates = sample_concepts(predicate_pool, k=k)

                    context = {
                        "task": task,
                        "subjects": subjects,
                        "objects": objects,
                        "predicates": predicates
                    }

                    response = call_llm(call=call, context=context, model=model)
                    response = response.get("response")
                    if isinstance(response, dict) and response:
                        propositions[minor_key].append({**response, "type": minor_key})
                        print(f"Success ({len(propositions[minor_key])} of {n} done): {response}\n")
                    else:
                        print(f"No valid sentence for task {minor_key} with current context: {context}")


    except KeyboardInterrupt:
        print("Process interrupted by user.")
    except Exception as e:
        print(f"Error during proposition generation: {e}")
        traceback.print_exc()

    with open("out/generated_propositions.json", "w") as f:
        json.dump(propositions, f, indent=4)
        print(f"Saved {len(propositions)} sets of {n} propositions to 'out/generated_propositions.json'")

    return propositions

def main(n:int=3):
    """
    Main function to generate n propositions per each task.
    """
    # Number of propositions to generate for each task
    generate_propositions(n=n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate semantically related proposition pairs based on annotations.")
    parser.add_argument("-n", type=int, default=3, help="Number of propositions to generate for each task (default: 3)")
    args = parser.parse_args()
    n = args.n if args.n > 0 else 3
    main(n=n)