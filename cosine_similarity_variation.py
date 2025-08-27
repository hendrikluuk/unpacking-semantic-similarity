#!/usr/bin/env python3
import json
import argparse
import pandas as pd
import numpy as np

from utils.models import models
from estimate_cosine_similarity import load_data

def norm(a: np.ndarray) -> np.ndarray:
    """
    Normalize the input array along the last axis.
    """
    return a / np.linalg.norm(a, axis=1, keepdims=True)

def main(output_file: str):
    """
    Report the min/max/mean/std of cosine similarities from concept-level and proposition-level data.
    Results are rounder to 3 significant digits.
    """
    results = []
    all_similarities = []  # Collect all individual similarity values for overall statistics

    running_min = 1.0
    running_max = 0.0
    min_entry = None
    max_entry = None

    for model in models:
        for dataset in ["propositions", "concepts"]:
            # Load embeddings for the {dataset}
            data = load_data(dataset, model, include_original_data=True)

            print(f"Processing model: {model}, dataset size x embedding size: {data['a'].shape}")
            a_norm = norm(data['a'])
            b_norm = norm(data['b'])
            # Element-wise multiplication followed by sum gives dot product between corresponding rows
            cosine_sim = np.sum(a_norm * b_norm, axis=1)

            # Collect all similarity values for overall statistics
            all_similarities.extend(cosine_sim.flatten().tolist())

            if cosine_sim.min() < running_min:
                running_min = cosine_sim.min()
                # get the index of the minimum value and locate the corresponding
                # entry from data['data']
                min_index = np.argmin(cosine_sim)
                min_entry = data['data'][min_index]
                print(f"New running min: {running_min} for model: {model}, dataset: {dataset}, entry:\n{json.dumps(min_entry, indent=4)}")

            if cosine_sim.max() > running_max:
                running_max = cosine_sim.max()
                max_index = np.argmax(cosine_sim)
                max_entry = data['data'][max_index]
                print(f"New running max: {running_max} for model: {model}, dataset: {dataset}, entry:\n{json.dumps(max_entry, indent=4)}")

            result = {
                'model': model,
                'dataset': dataset,
                'mean_similarity': round(float(cosine_sim.mean()), 3),
                'std_similarity': round(float(cosine_sim.std()), 3),
                'min_similarity': round(float(cosine_sim.min()), 3),
                'max_similarity': round(float(cosine_sim.max()), 3),
            }
            results.append(result)

    # Add overall statistics calculated from all individual similarity values
    all_similarities = pd.Series(all_similarities)
    overall_result = {
        'model': 'overall',
        'dataset': 'all',
        'mean_similarity': round(float(all_similarities.mean()), 3),
        'std_similarity': round(float(all_similarities.std()), 3),
        'min_similarity': round(float(all_similarities.min()), 3),
        'max_similarity': round(float(all_similarities.max()), 3),
        'min_entry': min_entry,
        'max_entry': max_entry,
    }

    results.append(overall_result)

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nOverall statistics: {json.dumps(overall_result, indent=4)}\n")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the variation in cosine similarity between text embeddings.")
    parser.add_argument("--output", type=str, default="out/cosine_similarity_variation.json",
                      help="Path to the output JSON file (default: out/cosine_similarity_variation.json)")
    
    args = parser.parse_args()
    main(args.output)
