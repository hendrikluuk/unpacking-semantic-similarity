#!/usr/bin/env python3
"""
  Load abstracts from a file (e.g., out/pubmed_disease_target_gene_therapeutic_top_1000.txt) and
  extract predicates from them using the Lumos AutoFx API. Submit 10 abstracts at a time as context.
"""
import os
import json
import argparse

from utils.call_llm import call_llm

call = "extract-predicates-from-abstract"
model = "gpt-4o-mini"

def load_abstracts(file_path:str="out/pubmed_disease_target_gene_therapeutic_top_10000.txt") -> list[str]:
    """
    Load abstracts from a file and return them as a list.
    Keep only lines starting with "Abstract: "
    """
    abstracts = []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("Abstract: "):
                abstracts.append(line.strip())
    return abstracts

def main(skip:int=0) -> None:
    """
    Main function to load abstracts and extract predicates.
    """
    # Load abstracts
    abstracts = load_abstracts()

    results = []

    try:
        # Process abstracts in batches of 10
        for i in range(skip, len(abstracts), 10):
            batch = abstracts[i:i+10]
            batch_no = i // 10 + 1 + skip
            context = {"abstracts": "\n\n".join(batch)}

            try:
                response = call_llm(call=call, context=context, model=model)
                response = response.get("response")
            except Exception as e:
                print(f"Error calling LLM (batch={batch_no}): {e}")
                response = []

            if isinstance(response, list) and response:
                results.append(response)
                print(f"Response ok for batch {batch_no}: extracted {len(response)} predicates")
            else:
                if isinstance(response, str):
                    print(f"Response error for batch {batch_no}: invalid JSON")
                else:
                    print(f"Error processing batch {batch_no}: {response}")
                results.append([])

    except KeyboardInterrupt:
        print("Process interrupted by user.")

    # append results to a json file 
    output_file = "out/extracted_predicates.json"
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_results = json.load(f)
            existing_results.extend(results)
            results = existing_results

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved {len(results)} results to '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract predicates from abstracts using Lumos AutoFx API.")
    parser.add_argument("--skip", type=int, default=0, help="Number of abstracts to skip.")
    args = parser.parse_args()
    main(skip=args.skip)