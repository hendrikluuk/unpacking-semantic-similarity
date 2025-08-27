#!/usr/bin/env python3
"""
Cluster predicates into relational domains
"""
import os
import json
import argparse

from utils.call_llm import call_llm

call = "identify-relational-domains"
model = "gpt-4.1"

def load_predicates(file_path: str = "out/extracted_predicates.json") -> list[list[dict]]:
    """
    Load predicates from a JSON file and return them as a list.
    """
    with open(file_path, "r") as f:
        predicates = json.load(f)
    return predicates

def main(skip: int = 0) -> None:
    """
    Main function to load predicates and cluster them into relational domains.
    """
    # Load predicates
    predicates = load_predicates()

    results = []

    try:
        # Process predicates in batches of 10
        for i in range(skip, len(predicates)):
            batch = predicates[i]
            context = {"predicates": json.dumps(batch), "state": json.dumps(results)}

            try:
                response = call_llm(call=call, context=context, model=model)
                response = response.get("response")
            except Exception as e:
                print(f"Error calling LLM (batch={i}): {e}")
                response = []

            if isinstance(response, list) and response:
                results.extend(response)
                print(f"Response ok for batch {i}: found {len(results)} relational domains")
            else:
                if isinstance(response, str):
                    print(f"Response error (invalid JSON) for batch {i}: {response}")
                else:
                    print(f"Error processing batch {i}: {response}")

    except KeyboardInterrupt:
        print("Process interrupted by user.")

    # Append results to a JSON file
    output_file = "out/clustered_predicates.json"
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_results = json.load(f)
            existing_results.extend(results)
            results = existing_results

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved {len(results)} results to '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster predicates into relational domains.")
    parser.add_argument("--skip", type=int, default=0, help="Number of batches to skip")
    args = parser.parse_args()

    main(skip=args.skip)