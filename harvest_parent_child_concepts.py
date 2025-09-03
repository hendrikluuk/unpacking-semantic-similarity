#!/usr/bin/env python3
"""
  Load concepts from the files matching the glob pattern '*.json' in 'probing_concepts/concepts' directory.
  Discard concepts where the attribute 'referents' is a list.

  Perform a depth-first traversal of the referents tree, recording all parent-child pairs of concepts.

  The referents tree looks like this:
    {
        "referents": {
            "nonacos-1-ene": {},
            "pentacos-1-ene": {},
            "2,4-dimethyl-1-heptene": {},
            "2-methyl-1-pentene": {},
            "ethene": {
                "(Z)-1,2-ethenediol": {}
            },
            "propene": {},
            "octadecene": {
                "octadec-9-ene": {
                    "cis-octadec-9-ene": {},
                    "trans-octadec-9-ene": {}
                },
                "octadec-1-ene": {},
                "octadec-7-ene": {
                    "trans-octadec-7-ene": {},
                    "cis-octadec-7-ene": {}
                },
                "octadec-2-ene": {}
            }, ...
    }

  Store each pair of parent-child concepts in the referents tree as an entry in the results in the following format:
    {
        "parent": {
            "label": "label1",
            "referents": ["referent1", "referent2"]
        },
        "child": {
            "label": "label2",
            "referents": ["referent3", "referent4"]
        },
        # use jarowinkler_similarity
        "label_similarity": jarowinkler_similarity(parent['label'], child['label']),
        "ss_sym": ss_sym(parent, child),
        "ss_asym_parent_child": ss_asym(parent, child),
        "ss_asym_child_parent": ss_asym(child, parent)
    }

    Note that you need to flatten the referents from the concept tree into a list of strings in the output.
"""
import os
import json
import glob
from jarowinkler import jarowinkler_similarity

from utils.similarity import ss_asym, ss_sym

max_semantic_field_size = 1000
min_semantic_similarity = 0.05
max_semantic_similarity = 0.95

def process_json_files(directory:str, pattern:str) -> list[dict]:
    """
    Load JSON files from a directory matching a given pattern.
    """
    files = glob.glob(os.path.join(directory, pattern))
    result = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            if not isinstance(data, dict):
                continue
            extract_referents(data, result)
    return result

def flatten(node:dict) -> list[str]:
    """
    Flatten the nodes in the referents tree.
    """
    result = []
    for label, tree in node.items():
        result.append(label)
        if isinstance(tree, dict) and tree:
            result.extend(flatten(tree))
    return list(set(result))

# extract the parent-child pairs from the concept tree of the given concept
def extract_referents(concept:dict, result:list[dict]=[], parent_label:str=None, domain:str='') -> list[dict]:
    """
    Extract referents from the concept tree.
    """
    if 'referents' in concept and not isinstance(concept['referents'], dict):
        # skip concepts with referents that are not in dict
        return

    if 'referents' in concept:
        parent_label = concept['concept']
        domain = concept['domain']
        concept = concept['referents']

    try:
        parent_referents = flatten(concept)
    except Exception as e:
        # if the referents are not a list, return
        print(f"'concept' must be a dict but got: {concept}")
        raise e

    if len(parent_referents) > max_semantic_field_size:
       # do not bother with concepts with large semantic fields
       return 

    for child_label, referents in concept.items():
        if not referents:
            child_referents = [child_label]
        else:
            child_referents = flatten(referents)
        r = {
            "domain": domain,
            "ss_sym": ss_sym(parent_referents, child_referents),
            "ss_asym_parent_child": ss_asym(parent_referents, child_referents),
            "ss_asym_child_parent": ss_asym(child_referents, parent_referents),
            "label_similarity": jarowinkler_similarity(parent_label, child_label)
        }
        r["parent"] = {
            "label": parent_label,
            "referents": parent_referents 
        }
        r["child"] = {
            "label": child_label,
            "referents": child_referents
        }
        if r["ss_sym"] >= min_semantic_similarity and r["ss_sym"] <= max_semantic_similarity:
            # only add the pair if the semantic similarity is in the range [min_semantic_similarity, 1.0)
            result.append(r)
        if referents:
            extract_referents(referents, result, child_label, domain)
    return result

def main():
    """
    Extract parent-child pairs of concepts.
    """
    directory = 'external_resources/probing_concepts/concepts'
    pattern = '*.json'
    results = process_json_files(directory, pattern)
    results = sorted(results, key=lambda x: x['ss_sym'])

    # Save the results to a file
    with open('out/parent_child_concepts.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Processed {len(results)} referents and saved to results.json")

if __name__ == "__main__":
    main()