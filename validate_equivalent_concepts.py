#!/usr/bin/env python3
"""
Validate equivalent concepts.

Load equivalent concepts from 'out/equivalent_concepts.json'
Present the LLM with both labels from the equivalent concept
and ask it to modify the labels to ensure that both labels are common enough
and they are unambigously referring to the same concept independently
of context.
"""
import json

from jarowinkler import jarowinkler_similarity

from utils.loaders import load_equivalent_concepts
from utils import call_llm

call = "validate-concept-labels"
model = "gpt-4.1"

def validate_concept_labels(concepts: list[dict]) -> list[dict]:
    """
    Validate equivalent concepts by ensuring that both labels are common enough
    and unambiguously refer to the same concept independently of context.
    """
    result = []

    for concept in concepts:
        context = {"input": json.dumps(concept)}

        try:
            response = call_llm(call=call, context=context, model=model)
            response = response.get("response")
        except Exception as e:
            print(f"Error calling LLM (input={context['input']}): {e}")
            response = {}
        
        if response:
            print(f"Original: {concept} \n Modified: {response}")
            result.append(response)
        else:
            result.append(concept)
            print(f"No modification needed for: {concept['label1']} = {concept['label2']}")
    return result

def main():
    """
    Main function to load equivalent concepts and validate them.
    """
    concepts = load_equivalent_concepts('out/equivalent_concepts.json')
    
    if not concepts:
        print("No concepts found.")
        return
    
    validated = validate_concept_labels(concepts)
    output_file = "out/validated_equivalent_concepts.json"

    # recalculate jaro winkler similarity
    for concept in validated:
        concept["jaro_winkler_similarity"] = jarowinkler_similarity(concept["label1"], concept["label2"])

    with open(output_file, "w") as f:
        json.dump(sorted(filter(lambda x: x["jaro_winkler_similarity"] < 1, validated), key=lambda x: x["jaro_winkler_similarity"]), f, indent=4)

    print(f"Validated {len(concepts)} equivalent concepts and saved {len(validated)} items to '{output_file}'")

if __name__ == "__main__":
    main()