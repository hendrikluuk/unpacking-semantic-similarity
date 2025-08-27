#!/usr/bin/env python3
"""
  Load training results and summarize performance metrics and estimate statistical significance.

  Example:
  ./summarize_train_results.py --pattern "trained_on_semantic_similarity/train_results*.json" --metric spearman_correlation

  # Single dataset
  ./summarize_train_results.py --datasets dataset1

  # Multiple datasets (space-separated)
  ./summarize_train_results.py --datasets dataset1 dataset2 dataset3

  # Multiple datasets with other arguments
  ./summarize_train_results.py --pattern "trained_on_semantic_similarity/train_results*.json" --datasets dataset1 dataset2 --metric spearman_correlation

  # No datasets (uses default empty list, includes all datasets)
  ./summarize_train_results.py --pattern "some_pattern.json"
"""
import os
import glob
import json
import argparse

import pandas as pd
from scipy.stats import binom, false_discovery_control

def summarize_train_results(results:list[dict], datasets:list[str]=[], performance_metric:str = 'pearson_correlation') -> pd.DataFrame:
    df = pd.DataFrame(results)

    if datasets:
        df = df[df['dataset'].isin(datasets)]
    # Sort by ascending 'mean' on each dataset and endpoint combination
    df = df.sort_values(by=['dataset', 'endpoint', 'out_dim', performance_metric], ascending=[True, True, False, False])
    # add rank to each group of dataset, endpoint and out_dim
    df['rank'] = df.groupby(['dataset', 'endpoint', 'out_dim'])[performance_metric].rank(method='first', ascending=False)
    # make an additional column of binary values indicating whether the rank was among the top third
    # assume that the number of models is divisible by 3
    df['top_third'] = df['rank'] <= len(df['model'].unique()) // 3

    # Reset index to have a clean DataFrame
    df.reset_index(drop=True, inplace=True)
    return df

def binom_cdf(k, n, p) -> float:
    """
    Return the cumulative probability of getting {k} or more successes
    from {n} trials with the probability of success {p} per trial.
    """
    # rounding helps the sum get to 1.0 for k=0
    return round(sum([binom.pmf(_k, n, p) for _k in range(k, n+1)]), 10)

def main(pattern:str, outfile:str, datasets:list[str], performance_metric:str):
    """
      Loop through .json files matching the pattern in the 'out' directory,
      concatenate their results, and summarize them.
    """
    # Find all JSON files matching the pattern
    json_files = glob.glob(pattern)
    if not json_files:
        print(f"No files found matching pattern: {pattern}")
        return

    # Load and concatenate results from all files
    all_results = []
    for file in json_files:
        with open(file, "r") as f:
            results = json.load(f)
            all_results.extend(results)    

    summary_df = summarize_train_results(all_results, datasets, performance_metric)

    # Save the summary to a XLSX file
    if not outfile:
        outfile = 'train_results_summary.xlsx'
    elif outfile.endswith('.json'):
        outfile = outfile.replace('.json', '.xlsx')

    # dataset and endpoint
    rank_summary = summary_df.groupby(['dataset', 'endpoint', 'model'])['rank'].sum().reset_index()
    rank_summary.rename(columns={'rank': 'sum_rank'}, inplace=True)
    # similarly to 'sum_rank', sum the values of 'top_third' for semantic endpoints
    rank_summary['sum_top_third'] = summary_df.groupby(['dataset', 'endpoint', 'model'])['top_third'].sum().reset_index(drop=True)
    # sort by dataset, endpoint and sum_rank
    rank_summary = rank_summary.sort_values(by=['dataset', 'endpoint', 'sum_rank'], ascending=True)

    number_of_summands = len(summary_df['endpoint'].unique()) // 2 * len(summary_df['out_dim'].unique()) * len(summary_df['dataset'].unique())

    # Calculate total rank sum for each model across specific endpoints only
    # Filter to include only semantic similarity measures (a -> b, b -> a)
    semantic_endpoints = ['a -> b', 'b -> a']
    filtered_ranks = rank_summary[rank_summary['endpoint'].isin(semantic_endpoints)]
    # Sum ranks of semantic endpoints for each model    
    semantic_ranks = filtered_ranks.groupby(['model'])['sum_rank'].sum().reset_index()
    # Sort by dataset and total rank (lower is better)
    semantic_ranks = semantic_ranks.sort_values(['sum_rank'], ascending=True)
    # add average rank for semantic endpoints
    semantic_ranks['avg_rank'] = semantic_ranks['sum_rank'] / number_of_summands
    # Sum 'sum_top_third' of semantic endpoints for each model
    semantic_ranks['sum_top_third'] = filtered_ranks.groupby(['model'])['sum_top_third'].sum().reset_index(drop=True)

    # Calculate the binomial probability of being in the top third 'sum_top_third' times out of the number of summands
    n = number_of_summands
    # probability of being in the top third, assuming uniform distribution of ranks
    p = 1 / 3 
    # The probability of the null hypothesis stating that the model's rank is randomly distributed is expressed as:
    # the cumulative probability of being in the top third {k} times out of {number_of_summands} tests
    semantic_ranks['p-binom'] = semantic_ranks['sum_top_third'].apply(
        lambda k: binom_cdf(k, n, p)
    )
    # using Benjamini-Yekuteli because some models might be derived from each
    # other and thus represent intrinsically correlated phenomena
    fdr = false_discovery_control(semantic_ranks["p-binom"], method="by")
    semantic_ranks['fdr'] = fdr

    # Sum ranks of symbolic similarity measures (char(a ~ b), token(a ~ b))
    symbolic_endpoints = ['char(a ~ b)', 'token(a ~ b)']
    filtered_ranks = rank_summary[rank_summary['endpoint'].isin(symbolic_endpoints)]
    symbolic_ranks = filtered_ranks.groupby(['model'])['sum_rank'].sum().reset_index()
    symbolic_ranks = symbolic_ranks.sort_values(['sum_rank'], ascending=True)
    symbolic_ranks['avg_rank'] = symbolic_ranks['sum_rank'] / number_of_summands
    symbolic_ranks['sum_top_third'] = filtered_ranks.groupby(['model'])['sum_top_third'].sum().reset_index(drop=True)
    symbolic_ranks['p-binom'] = symbolic_ranks['sum_top_third'].apply(
        lambda k: binom_cdf(k, n, p)
    )
    fdr = false_discovery_control(symbolic_ranks["p-binom"], method="by")
    symbolic_ranks['fdr'] = fdr

    with pd.ExcelWriter(outfile, engine='openpyxl', mode='w') as writer:
        summary_df.to_excel(writer, sheet_name='Train Results', index=False)
        rank_summary.to_excel(writer, sheet_name='Rank Summary', index=False)
        semantic_ranks.to_excel(writer, sheet_name='Semantic Ranking', index=False)
        symbolic_ranks.to_excel(writer, sheet_name='Symbolic Ranking', index=False)

    print(f"Summary saved to '{outfile}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize training results from JSON file.")
    parser.add_argument("--pattern", type=str, default="out/train_results*.json",
                        help="File name pattern for retrieving training results from ./out folder.")
    parser.add_argument("--outfile", type=str, default="",
                        help="Path to the output XLSX file for the summary.")
    parser.add_argument("--datasets", type=str, nargs='*', default=[],
                        help="List of datasets to filter results by. If empty, all datasets are included.")
    parser.add_argument("--metric", type=str, default="pearson_correlation",
                        help="Performance metric to use for ranking models (default: 'pearson_correlation'). Options: 'pearson_correlation', 'spearman_correlation'.")
    args = parser.parse_args()
    main(pattern=args.pattern, outfile=args.outfile, datasets=args.datasets, performance_metric=args.metric)