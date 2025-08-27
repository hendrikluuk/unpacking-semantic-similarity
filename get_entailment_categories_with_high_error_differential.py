#!/usr/bin/env python3
"""
  Load data about the divergence of semantic entailment estimates per domain between a pair of models from out/model_comparison.json.
  In each comparison rank significant domains by fold-difference and record for each domain whether it was in the top third of the ranks.
  Use the rationale from summarize_train_results.py to get a statistical estimate of how likely the observed number of top third
  ranks is to occur under the null hypothesis that domain ranks are distributed uniformly.
"""
import json
import argparse

import pandas as pd

from compare_deltas import FOLD_DIFFERENCE_KEY
from summarize_train_results import binom_cdf, false_discovery_control 

def load_data(p:float, file_path:str='out/prediction_error_comparison.json') -> tuple[list[dict], list[dict], int]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    result = []
    means = []
    n = len(data)
    for comparison in data:
        # Sort domains by fold-difference
        comparison['comparisons'].sort(key=lambda x: x[FOLD_DIFFERENCE_KEY], reverse=True)
        
        # Get the rank of each domain based on fold-difference
        for i, domain in enumerate(comparison['comparisons']):
            domain['rank'] = i + 1
        
        # Determine top third domains
        total_domains = len(comparison['comparisons'])
        top_fraction_count = int(total_domains * p)

        for domain in comparison['comparisons']:
            means.append({
                'domain': domain['domain'],
                'mean estimate (model1)': domain['mean_model1'],
                'mean estimate (model2)': domain['mean_model2'],
                'mean error (model1)': domain['mean_delta_model1'],
                'mean error (model2)': domain['mean_delta_model2']
            })
            domain['top_fraction'] = domain['rank'] <= top_fraction_count and domain.get('significant', False)

        result.extend(map(lambda x: normalize_item(x, comparison['model1_name'], comparison['model2_name']), comparison['comparisons']))   
    return result, means, n

def normalize_item(comparison:dict, model1:str, model2:str) -> dict:
    return {
        'model1': model1,
        'model2': model2,
        'domain': comparison['domain'],
        'rank': comparison['rank'],
        'top_fraction': comparison['top_fraction']
    }

def main(p:float=1/3):
    """
    p is the probability of ranking to the top fraction based on the null hypothesis
    p = 1/3 is the top third
    p = 1/2 is the top half
    p = 1/4 is the top quarter etc
    """
    data, means, n = load_data(p)

    # get the count of 'top_fraction' for each domain across all model comparisons using Pandas
    df = pd.DataFrame(data)
    summary = df.groupby('domain')['top_fraction'].sum().reset_index()
    summary['p-binom'] = summary['top_fraction'].apply(lambda k: binom_cdf(k, n, p))
    summary['q'] = false_discovery_control(summary["p-binom"], method="by")
    summary['significant'] = summary['q'] < 0.05
    summary = summary.sort_values(by='p-binom', ascending=True)

    means = pd.DataFrame(means)
    means = means.groupby('domain')[['mean estimate (model1)', 'mean estimate (model2)', 'mean error (model1)', 'mean error (model2)']].mean().reset_index()
    summary = summary.merge(means, on='domain')

    outfile = 'out/entailment_categories_with_differential_error_significance.csv'
    summary.to_csv(outfile, index=False)
    print(summary)
    print(f"Summary saved to '{outfile}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify proposition construction schemas where the estimates of semantic entailment are most divergent between the compared models.")
    parser.add_argument('--p', type=float, default=1/3, help="Probability threshold for top fraction (default: 1/3 for top third)")
    args = parser.parse_args()
    main(args.p)