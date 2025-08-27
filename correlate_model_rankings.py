#!/usr/bin/env python
"""
  Correlate performance rankings in the estimation of semantic and symbolic similarity 
  in different runs.

  Example usage:
  ./correlate_rankings.py --base-dir "latest_runs"
"""

import os
import argparse

import pandas as pd
from scipy.stats import spearmanr

from utils.models import models

# Set pandas display options to 3 significant digits
pd.set_option('display.float_format', lambda x: f'{x:.3g}')

mteb_models = [{'model': key, 'mteb_rank': model['mteb_rank']} for key, model in models.items()]

def correlate_rankings_for_run(run_number:int, base_dir:str=".") -> dict:
    """
    Calculate Spearman correlation between semantic and symbolic rankings for a specific run.
    
    Args:
        run_number (int): The run number (1, 2, or 3)
        base_dir (str): Base directory containing the train_run folders
        
    Returns:
        tuple: (correlation_coefficient, p_value) or (None, None) if error
    """
    # Determine file paths
    run_dir = os.path.join(base_dir, f"run{run_number}")
    excel_file = os.path.join(run_dir, f"train_results_summary_run{run_number}.xlsx")
    
    print(f"Processing run {run_number}: {excel_file}")
    
    # Load semantic and symbolic rankings
    semantic_df = pd.read_excel(excel_file, sheet_name='Semantic Ranking').sort_values('model')
    symbolic_df = pd.read_excel(excel_file, sheet_name='Symbolic Ranking').sort_values('model')
    
    # Calculate Spearman correlation
    correlation, p_value = spearmanr(semantic_df['avg_rank'], symbolic_df['avg_rank'])

    # Calculate Spearman correlation of both rankings with MTEB rankings
    mteb_df = pd.DataFrame(mteb_models)
    mteb_df = mteb_df[mteb_df['mteb_rank'].notna()]
    semantic_mteb_df = semantic_df.merge(mteb_df, left_on='model', right_on='model', how='inner')
    symbolic_mteb_df = symbolic_df.merge(mteb_df, left_on='model', right_on='model', how='inner')
    semantic_correlation, semantic_p_value = spearmanr(semantic_mteb_df['avg_rank'], semantic_mteb_df['mteb_rank'])
    symbolic_correlation, symbolic_p_value = spearmanr(symbolic_mteb_df['avg_rank'], symbolic_mteb_df['mteb_rank'])

    return [
        {'run': run_number, 'comparison': 'semantic vs symbolic', 'spearman rho': correlation, 'p': p_value},
        {'run': run_number, 'comparison': 'semantic vs MTEB', 'spearman rho': semantic_correlation, 'p': semantic_p_value},
        {'run': run_number, 'comparison': 'symbolic vs MTEB', 'spearman rho': symbolic_correlation, 'p': symbolic_p_value},
    ]

def main(base_dir: str, runs: list):
    """Main function to execute the correlation analysis."""
    print("Correlating semantic and symbolic similarity rankings")
    print("=" * 60)
    
    results = []
    
    for run_num in runs:
        print(f"\nAnalyzing Run {run_num}:")
        print("-" * 20)
        
        correlations = correlate_rankings_for_run(run_num, base_dir)
        
        results.extend(correlations)
    
    # Create a DataFrame to summarize results
    summary_df = pd.DataFrame(results)
    summary_df.set_index('run', inplace=True)
    print(f"\nCorrelation Summary:\n{summary_df}")

    # calculate average correlation and p-value for each comparison across runs
    avg_summary = summary_df.groupby('comparison').agg(
        {'spearman rho': 'mean', 'p': 'mean'}
    ).reset_index()
    avg_summary['run'] = 'average'
    avg_summary.set_index('run', inplace=True)

    print(f"\nAverage Correlation Summary:\n{avg_summary}")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Correlate semantic and symbolic similarity rankings across runs.")
    arg_parser.add_argument("--base-dir", type=str, default=".",
                            help="Base directory containing train_run folders (default: current directory)")
    arg_parser.add_argument("--runs", type=int, nargs='+', default=[1, 2, 3],
                            help="Specific run numbers to analyze (default: 1 2 3)")
    args = arg_parser.parse_args()
    main(args.base_dir, args.runs)