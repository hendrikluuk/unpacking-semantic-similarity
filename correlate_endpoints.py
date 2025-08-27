#!/usr/bin/env python3
"""
Load 'out/sampled_concepts_2500.json' and 'out/sampled_propositions_1500.json'
and correlate different endpoints within each dataset using Pearson and Spearman correlations.
"""
import json
import argparse
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr

endpoints = ["a -> b", "b -> a", "char(a ~ b)", "token(a ~ b)"]

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the JSON data from the specified file path and return it as a DataFrame.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def correlate_endpoints(df: pd.DataFrame, dataset:str) -> pd.DataFrame:
    """
    Calculate Pearson and Spearman correlations for each pair of endpoints.
    """
    global endpoints
    correlations = []
    
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            endpoint1 = endpoints[i]
            endpoint2 = endpoints[j]
            pearson_corr, pearson_p = pearsonr(df[endpoint1], df[endpoint2])
            spearman_corr, spearman_p = spearmanr(df[endpoint1], df[endpoint2])
            
            correlations.append({
                'dataset': dataset,
                'endpoint1': endpoint1,
                'endpoint2': endpoint2,
                'pearson_corr': pearson_corr,
                'pearson_p': pearson_p,
                'spearman_corr': spearman_corr,
                'spearman_p': spearman_p
            })
    
    return pd.DataFrame(correlations)

def main(concepts_file: str, propositions_file: str):
    """
    Main function to load data, calculate correlations, and print results.
    """
    concepts_df = load_data(concepts_file)
    propositions_df = load_data(propositions_file)

    print("Correlating concepts...")
    concepts_corr = correlate_endpoints(concepts_df, "concepts")
    print(concepts_corr)

    print("\nCorrelating propositions...")
    propositions_corr = correlate_endpoints(propositions_df, "propositions")
    print(propositions_corr)

    result = pd.concat([concepts_corr, propositions_corr], ignore_index=True)
    outfile = 'out/endpoint_correlations.csv'
    result.to_csv(outfile, index=False)
    print(f"Results saved to {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correlate endpoints in concepts and propositions datasets.")
    parser.add_argument('--concepts-file', type=str, default='out/sampled_concepts_2500.json', help='Path to concepts JSON file')
    parser.add_argument('--propositions-file', type=str, default='out/sampled_propositions_1500.json', help='Path to propositions JSON file')
    
    args = parser.parse_args()
    main(args.concepts_file, args.propositions_file)