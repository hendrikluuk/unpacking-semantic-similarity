#!/usr/bin/env python
"""
  Load responses from the files matching the glob pattern '*limited-list-referents*.json' in probing_concepts/scores directory.
  For each file, extract the ['judgement'][matches'] key and store the values in a list.
  In this list, locate all entries that contain a name and another name in parentheses (e.g., "CCAAT/enhancer-binding protein alpha (CEBPA)", "NFKB1 (NFKB1)")
  Discard entries where the names are the same.
  Parse remaining entries to extract the name and the name in parentheses and calculate jaro winkler similarity between them.
  Store the pair in a dictionary with the form:
    {
        "label1": "CCAAT/enhancer-binding protein alpha",
        "label2": "CEBPA",
        "jaro_winkler_similarity": 0.8,
        "semantic_similarity": 1.0
    }

  Use the rapidfuzz JaroWinkler package to calculate the Jaro-Winkler similarity:
    from jarowinkler import *
    jaro_similarity(entry['label1'], entry['label2'])

  Store all pairs in a list, sort the list by jaro_winkler_similarity in ascending order, and save the list to a JSON file with the name 'identical_concepts.json' in the same directory.
"""
import os
import json
import glob
from jarowinkler import jarowinkler_similarity

trusted_models = set([
    'gpt-4o',
    'anthropic.claude-3-5-sonnet-v1:0',
])

def load_json_files(directory:str, pattern:str, models:list[str]|None=trusted_models) -> list[dict]:
    """
    Load JSON files from a directory matching a given pattern.
    """
    files = glob.glob(os.path.join(directory, pattern))
    result = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            for response in data:
                if models and response['responder'] not in models:
                    continue
                result.append(response)
    return result 

def extract_matches(data:list[dict]) -> list[tuple[str,str]]:
    """
    Extract matches from the data.
    """
    matches = []
    for entry in data:
        if 'judgement' in entry and 'matches' in entry['judgement']:
            for match in entry['judgement']['matches']:
               matches.append((entry['domain'], entry['responder'], match))
    return matches

def filter_and_parse_matches(matches: list[tuple[str,str]]) -> list[dict]:
    """
    Filter and parse matches to extract relevant information.
    """
    processed = set()
    filtered = []
    for domain, responder, match in matches:
        if isinstance(match, str) and " (" in match: # Add check for delimiter
            # Split the match into label1 and label2
            try:
                label1, label2 = match.split(" (")
            except ValueError:
                print(f"Warning: error splitting match: {match}. Skipping.")
                continue
            label2 = label2.rstrip(")")
            if label1 != label2:
                # Calculate Jaro-Winkler similarity
                similarity = jarowinkler_similarity(label1, label2)
                # Check if the pair has been processed before by at most one other model
                h = hash((label1.lower(), label2.lower(), responder))
                if h in processed:
                    # if the pair has been processed by the same model, skip it
                    continue
                if sum([hash((label1.lower(), label2.lower(), model)) in processed for model in trusted_models.difference(set([responder]))]) == 1:
                    # add only pairs that another model has already suggested
                    filtered.append({
                        "domain": domain,
                        "label1": label1,
                        "label2": label2,
                        "jaro_winkler_similarity": similarity
                    })
                processed.add(h)
    return filtered

def save_to_json(data: list[dict], filename: str):
    """
    Save data to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    data = load_json_files('probing_concepts/scores', '*limited-list-referents*.json')
    matches = extract_matches(data)
    filtered = filter_and_parse_matches(matches)
    save_to_json(sorted(filtered, key=lambda x: x['jaro_winkler_similarity']), 'out/equivalent_concepts.json')

if __name__ == "__main__":
    main()