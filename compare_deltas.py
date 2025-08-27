#!/usr/bin/env python3
"""
Compare prediction deltas between two models on the propositions dataset.

This script takes two model names as input, loads both models trained on propositions 
to estimate the 'a -> b' endpoint, and loads the sampled_propositions_1500.json dataset.

For each model, it generates similarity predictions for all items in the dataset and 
calculates absolute deviations (abs_delta) from the ground truth 'a -> b' values.

Then it compares the deltas between the two models within each domain using appropriate 
statistical tests (paired t-test or Wilcoxon signed-rank test) and applies multiple 
testing correction (FDR - False Discovery Rate) to control for multiple comparisons 
across domains.

# Basic usage
./compare_deltas.py bge-m3 gte-large-en-v1.5

# With custom output file
./compare_deltas.py bge-m3 gte-large-en-v1.5 --output out/model_comparison.json
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ml.similarity_learner import SimilarityLearner
from estimate_cosine_similarity import load_data

FOLD_DIFFERENCE_KEY = 'mean_delta_model2 / mean_delta_model1'

def calculate_deltas(predictions: list[float], ground_truth: list[float]) -> np.ndarray:
    """Calculate absolute deviations between predictions and ground truth."""
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    return np.abs(predictions - ground_truth)

def load_model_predictions(model_name: str, data: dict) -> dict:
    """
    Load model and generate predictions for the given data.
    
    Args:
        model_name: Name of the model to load
        data: Dataset with embeddings and ground truth
    
    Returns:
        Dictionary with predictions
    """
    # Find the model file
    model_pattern = f"models/{model_name}_propositions_a->b_*.json"
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        raise FileNotFoundError(f"No model files found matching pattern: {model_pattern}")
    
    results = {}
    # generate prediction from all projection sizes
    for model_file in model_files:
        learner = SimilarityLearner().load(model_file)
        
        predictions = learner.predict(data['a'], data['b'], data['y'], include_baseline=False)

        # Calculate average prediction for each domain
        df = pd.DataFrame(data['data'])

        df['predicted'] = predictions['projected']
        df['delta'] = calculate_deltas(predictions['projected'], data['y'].flatten())

        # Group by domain and calculate mean predictions
        df_grouped = df.groupby('domain')['predicted'].mean().reset_index()
        domain_means = df_grouped.set_index('domain')['predicted'].to_dict()

        # Calculate absolute deltas
        
        out_dim = learner.config.out_dim
        results[out_dim] = {
            'model_name': model_name,
            'model_file': model_file,
            'out_dim': out_dim,
            'deltas': df,
            'domain_means': domain_means,
            'model_performance': {
                'mse': predictions['projected_mse'],
                'mae': predictions['projected_mae'],
                'pearson_correlation': predictions['projected_pearson'],
                'spearman_correlation': predictions['projected_spearman']
            }
        }
    
    return results

def perform_paired_comparison(deltas1: np.ndarray, deltas2: np.ndarray) -> dict:
    """
    Perform paired statistical test comparing deltas between two models for a specific domain.
    
    Args:
        deltas1: Absolute deltas from model 1
        deltas2: Absolute deltas from model 2 
        domain: Domain name for reference
    
    Returns:
        Dictionary with test results
    """
    assert(deltas1.size == deltas2.size)
    
    # Calculate basic statistics
    mean_delta1 = float(np.mean(deltas1))
    mean_delta2 = float(np.mean(deltas2))
    
    fold_difference = mean_delta2 / mean_delta1
    
    statistic, p_value = stats.wilcoxon(deltas1, deltas2, alternative='two-sided')
    return {
        'domain': '',   # will be filled by the caller
        'sample_size': deltas1.size,
        'mean_model1': 0.0,   # will be filled by the caller
        'mean_model2': 0.0,   # will be filled by the caller
        'mean_delta_model1': mean_delta1,
        'mean_delta_model2': mean_delta2,
        FOLD_DIFFERENCE_KEY: fold_difference,
        'test_statistic': float(statistic) if not np.isnan(statistic) else None,
        'p_value': float(p_value),
    }

def apply_multiple_testing_correction(comparison_results: list, method: str = 'fdr_by') -> list[dict]:
    """
    Apply multiple testing correction to p-values.
    
    Args:
        comparison_results: List of comparison result dictionaries
        method: Multiple testing correction method (default: 'fdr_by' for Benjamini-Yekutieli FDR)
    
    Returns:
        Sorted comparison results with added q-values
    """
    # Extract p-values
    p_values = [result['p_value'] for result in comparison_results]
    
    # Apply multiple testing correction
    reject, p_corrected, _, _ = multipletests(p_values, method=method)
    
    # Update results with corrected p-values and significance
    for i, result in enumerate(comparison_results):
        result['q_value'] = float(p_corrected[i])
        result['significant'] = bool(reject[i])
    
    return sorted(comparison_results, key=lambda x: x['q_value'])

def print_domain_definitions():
    """
    Print domain definitions from annotations/proposition_similarity_constraints.json
    in alphabetical order by domain ID.
    """
    try:
        constraints_file = "annotations/proposition_similarity_constraints.json"
        with open(constraints_file, 'r') as f:
            constraints = json.load(f)
        
        validation_domains = constraints.get("for validation", {})
        
        if not validation_domains:
            print("\nWarning: No validation domains found in constraints file")
            return
        
        print("\n" + "="*80)
        print("SIMILARITY DOMAIN DEFINITIONS")
        print("="*80)
        
        # Sort domains alphabetically by domain ID
        sorted_domains = sorted(validation_domains.items(), key=lambda x: x[0])
        
        for domain_id, domain_info in sorted_domains:
            comment = domain_info.get('comment', 'No description available')
            subject = domain_info.get('subject', '?')
            object_type = domain_info.get('object', '?')
            predicate = domain_info.get('predicate', '?')
            
            print(f"\nDomain {domain_id}:")
            print(f"  Description: {comment}")
            print(f"  Pattern: Subject={subject}, Object={object_type}, Predicate={predicate}")
            
            # Add legend for the pattern codes
            if domain_id == sorted_domains[0][0]:  # Only show legend for first domain
                print(f"\n  Legend: I=Identical, S=Synonymous, N=Narrower/broader, C=Contradictory, U=Unrelated")
        
        print("\n" + "="*80)
        
    except FileNotFoundError:
        print(f"\nWarning: Could not find {constraints_file}")
    except json.JSONDecodeError:
        print(f"\nWarning: Could not parse {constraints_file}")
    except Exception as e:
        print(f"\nWarning: Error loading domain definitions: {e}")

def main(
        model1_name: str,
        model2_name: str,
        output_file: str,
        fdr_method: str,
        overwrite: bool = False
    ):
    """
    Main function to compare deltas between two models.
    
    Args:
        model1_name: Name of the first model to compare
        model2_name: Name of the second model to compare
        output_file: Path to the output JSON file
    """
    print(f"Starting delta comparison between models: {model1_name} vs {model2_name}")
    
    # Load dataset with embeddings (using model1_name for embeddings, assuming same embeddings for both)
    data1 = load_data("propositions", model1_name, include_original_data=True)
    data2 = load_data("propositions", model2_name, include_original_data=True)

    domains = [item['domain'] for item in data1['data']]
    unique_domains = sorted(set(domains))
    items_per_domain = {domain: domains.count(domain) for domain in unique_domains}
    
    # Load predictions from both models
    print(f"Loading predictions from {model1_name}...")
    model1 = load_model_predictions(model1_name, data1)
    
    print(f"Loading predictions from {model2_name}...")
    model2 = load_model_predictions(model2_name, data2)
    
    # Get available output dimensions
    dims = sorted(model1.keys())
    
    # Assemble deltas for each {domain} x {out_dim}
    dim_results = {}
    
    for out_dim in dims:
        comparison_results = []
        for domain in unique_domains:
            model1_result = model1[out_dim]
            model2_result = model2[out_dim]
            # index is the same for both models as they are evaluate on the same dataset
            domain_index = model1_result['deltas']['domain'] == domain

            model1_deltas = model1_result['deltas'].loc[domain_index, 'delta']
            model2_deltas = model2_result['deltas'].loc[domain_index, 'delta']

            # Perform single statistical comparison for this domain
            comparison = perform_paired_comparison(model1_deltas, model2_deltas)
            comparison['domain'] = domain
            comparison['mean_model1'] = model1_result['domain_means'][domain]
            comparison['mean_model2'] = model2_result['domain_means'][domain]
            
            comparison_results.append(comparison)

        dim_results[out_dim] = {
            'out_dim': out_dim,
            'comparisons': comparison_results,
            'model1_file': model1[out_dim]['model_file'],
            'model2_file': model2[out_dim]['model_file'],
            'model1_name': model1_name,
            'model2_name': model2_name,
        }

    # group results by out_dim
    final_results = []    
    # Apply multiple testing correction across domains
    for out_dim, dim_result in dim_results.items():
        comparisons = apply_multiple_testing_correction(dim_result['comparisons'], method=fdr_method)

        # Prepare final results
        final_results.append({
            'comparison_type': 'Wilcoxon rank-sum test to compare semantic entailment estimation error in each proposition category between two models',
            'dataset': 'propositions',
            'model1_name': model1_name,
            'model2_name': model2_name,\
            'projection_dimension': out_dim,
            'samples_per_proposition_category': items_per_domain,
            'total_proposition_categories_compared': len(comparison_results),
            'significant_proposition_categories': sum(r['significant'] for r in comparisons),
            'fdr_correction_method': fdr_method,
            'model_files': {
                'model1': dim_result['model1_file'],
                'model2': dim_result['model2_file']
            },
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'comparisons': comparisons 
        })
    
    # Load existing results if output file exists
    existing_results = []
    if os.path.exists(output_file) and not overwrite:
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_results = []
    
    # Append new results
    existing_results.extend(final_results)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(existing_results, f, indent=2)
    
    # Print domain definitions at the end
    print_domain_definitions()
    print(f"Analysis complete. Results saved to '{output_file}'")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare prediction deltas between two models.")
    parser.add_argument("model1_name", type=str,
                      help="Name of the first model to compare (e.g., 'bge-m3')")
    parser.add_argument("model2_name", type=str,
                      help="Name of the second model to compare (e.g., 'gte-large-en-v1.5')")
    parser.add_argument("--output", type=str, default="out/prediction_error_comparison.json",
                      help="Path to the output JSON file (default: out/prediction_error_comparison.json)")
    parser.add_argument("--fdr-method", type=str, default='fdr_by',
                      help="Method for multiple testing correction (default: 'fdr_by')")
    parser.add_argument("--overwrite", action='store_true',
                      help="Overwrite existing output file if it exists (default: False)")
    
    args = parser.parse_args()
    main(args.model1_name, args.model2_name, args.output, args.fdr_method, args.overwrite)
