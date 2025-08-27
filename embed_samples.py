#!/usr/bin/env python3
"""
  Embed semantic similarity samples using available models.

  Example usage:
  ./embed_samples.py --pattern "out/sampled_*.json"
  ./embed_samples.py --pattern "out/scitail_*.json"
"""
import os
import glob
import json
import argparse
import traceback

from utils.embedder import Embedder
from utils.models import models

def download_models():
    for model in models:
        try:
            e = Embedder(model, local_files_only=False)
            print(f"Successfully downloaded model: {model}")
        except Exception as e:
            print(f"Failed to download model {model}: {e}")


def main(filepattern:str, download: bool = False, rebuild: bool = False):
    if download:
        print("Downloading models...")
        download_models()
        print("Models downloaded successfully.")

    filenames = glob.glob(filepattern)
    if not filenames:
        print(f"No files found matching pattern: {filepattern}")
        return
    
    for filename in filenames:
        if not os.path.exists(filename):
            print(f"File {filename} does not exist. Skipping.")
            continue

        with open(filename, 'r') as f:
            concepts = json.load(f)

        for model in models:
            if "concepts" in filename:
                tag = "concepts"
            elif "propositions" in filename:
                tag = "propositions"
            else:
                # assign file name without path and extension as tag
                tag = os.path.splitext(os.path.basename(filename))[0]

            cache_file = f"cache/{model}/{tag}.pkl"
            if os.path.exists(cache_file) and not rebuild:
                print(f"Cache file exists for'{model}'. Skipping...")
                continue
            try:
                print(f"Indexing data in '{filename}' with model: '{model}'")
                embedder = Embedder(model=model, cache_tag=tag, local_files_only=True)
                embedder.build_index(concepts, rebuild=rebuild)
            except KeyboardInterrupt:
                print("Process interrupted by user.")
                user_input = input("Do you want to abort (a) or continue (c)? ").strip().lower()
                if user_input == 'a':
                    print("Aborting...")
                    return
                print("Continuing...")
                continue
            except Exception as e:
                print(f"Error with model {model} on file '{filename}': {e}")
                traceback.print_exc()
                print("Continuing with the next model...")
                continue

            # free memory for the next model
            del embedder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed sampled data with available models.")

    # e.g. out/sampled_* to cover ["out/sampled_concepts_2500.json", "out/sampled_propositions_1500.json"]
    parser.add_argument("--pattern", type=str, default="out/sampled_*.json", help="Pattern to match files to embed. Default is 'out/sampled_*.json'.")

    parser.add_argument("--download", action="store_true", help="Download models from Hugging Face.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index for existing models.")
    args = parser.parse_args()
    main(args.pattern, args.download, args.rebuild)
