#!/usr/bin/env python3
"""
    Perform regression analysis of cosine similarities in 'out/cosine_similarity_trial.json'.
"""
import os
import json
import argparse
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from utils.models import models

# Define features for regression
# Will exclude 'b -> a', because it is identical with 'a -> b'
global_feature_columns = ['a -> b', 'char(a ~ b)', 'token(a ~ b)', 'context switch']

def load_validation_results(file_path: str) -> List[Dict[str, Any]]:
    """Load validation results from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load the corresponding dataset from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def perform_regression_analysis(df: pd.DataFrame, all_models: List[str], absolute_values:bool=False, drop_entailment:bool=False, standardize:bool=True) -> Dict[str, Any]:
    """
    Perform linear regression analysis for each model using the specified feature columns.
    
    Args:
        df: DataFrame containing the data
        all_models: List of model names to analyze
        absolute_values: Should we convert all features to absolute values?
        drop_entailment: Should we ignore semantic entailment features?
        standardize: Should we standardize (normalize) features to mean=0, std=1? Default True
    
    Returns:
        Dictionary containing regression statistics for each model
    """
    if drop_entailment:
        # See what happens if we drop the entailment similarity measure ('a -> b')
        feature_columns = global_feature_columns[1:]
    else:
        feature_columns = global_feature_columns
    # Prepare features and target
    features = df[feature_columns].values
    if absolute_values:
        # Convert all features to absolute values. Will only affect entailment similarity 
        # measures ('a -> b', 'b -> a') since the remaining features are never negative.
        features = np.abs(features)

    regression_stats = {}  # Store regression statistics for each model
    for model in all_models:
        # Initialize relative_contributions for this model
        relative_contributions = {}
        
        target = df[model].values
        
        # Remove rows with NaN values
        valid_indices = ~(np.isnan(features).any(axis=1) | np.isnan(target))
        if valid_indices.sum() <= len(feature_columns) + 1:  # Need more data points than features
            continue  # Skip if insufficient valid data
            
        X = features[valid_indices]
        y = target[valid_indices]
        
        # Standardize features if requested (default: True)
        if standardize:
            X = StandardScaler().fit_transform(X)
        
        # Create and fit the linear regression model
        reg_model = LinearRegression()
        reg_model.fit(X, y)
        
        # Make predictions for all rows (including those used for training)
        predictions = np.full(len(df), np.nan)
        # Use the already transformed X for predictions
        y_pred = reg_model.predict(X)
        predictions[valid_indices] = y_pred
        
        # Calculate statistical significance
        n = len(y)  # number of observations
        p = len(feature_columns)  # number of features
        
        # Calculate residuals and standard errors
        residuals = y - y_pred
        normalization_factor = n - p - 1 if n - p - 1 > 0 else 1
        mse = np.sum(residuals**2) / normalization_factor  # Mean squared error with degrees of freedom correction
        
        # Calculate R-squared and adjusted R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2_score = 1 - (ss_res / ss_tot)
        adj_r2 = 1 - (1 - r2_score) * (n - 1) / normalization_factor
        
        # F-statistic for overall model significance
        f_stat = (r2_score / p) / ((1 - r2_score) / normalization_factor)
        f_pvalue = 1 - stats.f.cdf(f_stat, p, normalization_factor)
        
        # Calculate standard errors for coefficients
        # X^T * X inverse for coefficient standard errors
        X_with_intercept = np.column_stack([np.ones(n), X])  # Add intercept column
        xtx_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        se_coef = np.sqrt(np.diagonal(xtx_inv) * mse)
        
        # T-statistics and p-values for each coefficient
        coef_with_intercept = np.concatenate([[reg_model.intercept_], reg_model.coef_])
        t_stats = coef_with_intercept / se_coef
        # Two-tailed p-values for coefficients
        coef_pvalues = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
        
        # Calculate relative contributions for standardized coefficients
        if standardize:
            abs_coefs = np.abs(reg_model.coef_)
            total_abs_coef = np.sum(abs_coefs)
            if total_abs_coef > 0:
                relative_contributions = {
                    feature: (abs(coef) / total_abs_coef) * 100 
                    for feature, coef in zip(feature_columns, reg_model.coef_)
                }
        
        # Store regression statistics
        regression_stats[model] = {
            'R2': r2_score,
            'adj_R2': adj_r2,
            'f_stat': f_stat,
            'f_pvalue': f_pvalue,
            'intercept': reg_model.intercept_,
            'intercept_pvalue': coef_pvalues[0],
            'coefficients': dict(zip(feature_columns, reg_model.coef_)),
            'coef_pvalues': dict(zip(feature_columns, coef_pvalues[1:])),
            'coef_se': dict(zip(['intercept'] + feature_columns, se_coef)),
            'relative_contributions': relative_contributions if standardize else {},
            'n': n,
            'standardized': standardize
        }
        
        # Print detailed statistics
        print(f"\nModel: {model}")
        if standardize:
            print(f"  (Using standardized features)")
        print(f"  R² = {r2_score:.4f}, Adjusted R² = {adj_r2:.4f}")
        print(f"  F-statistic = {f_stat:.4f}, F p-value = {f_pvalue:.6f}")
        print(f"  Intercept = {reg_model.intercept_:.4f} (p = {coef_pvalues[0]:.6f})")
        print("  Coefficients:")
        for i, feature in enumerate(feature_columns):
            coef = reg_model.coef_[i]
            pval = coef_pvalues[i+1]
            significance = get_sig(pval) 
            coef_label = f"{feature} (standardized)" if standardize else feature
            
            if standardize and feature in relative_contributions:
                rel_contrib = relative_contributions[feature]
                print(f"    {coef_label}: {coef:.4f} (p = {pval:.6f}) {significance} [{rel_contrib:.1f}%]")
            else:
                print(f"    {coef_label}: {coef:.4f} (p = {pval:.6f}) {significance}")

    return regression_stats

def expand_feature_set(X_base: pd.DataFrame, base_feature_columns, threshold: float = 0.95) -> pd.DataFrame:
    """
    Expand the feature set by adding polynomial features and interaction terms.
    Remove multicollinear features based on the specified correlation threshold.
    """
    # Create polynomial features (degree=2) to get interaction terms
    # This will create: original features + pairwise interactions + square terms
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X = poly.fit_transform(X_base)

    # Get feature names for the polynomial features
    feature_columns = list(poly.get_feature_names_out(base_feature_columns))

    # Remove features with zero variance (constant features)
    feature_variances = np.var(X, axis=0)
    non_zero_var_indices = feature_variances > 1e-10  # Use small threshold to handle numerical precision
    X = X[:, non_zero_var_indices]
    feature_columns = [feature_columns[i] for i in range(len(feature_columns)) if non_zero_var_indices[i]]
    
    if len(feature_columns) != len(non_zero_var_indices):
        removed_zero_var = len(non_zero_var_indices) - len(feature_columns)
        print(f"Removed {removed_zero_var} features with zero variance")
    
    # Report features before multicollinearity removal
    n_features_before_multicol = len(feature_columns)
    print(f"Features before multicollinearity removal: {n_features_before_multicol}")
    
    # Remove multicollinear features using the existing function
    # NB! removed features will not be included in feature_columns and report
    X, feature_columns = remove_multicollinear_features(X, feature_columns, threshold=threshold)
    
    # Report features after multicollinearity removal
    n_features_after_multicol = len(feature_columns)
    print(f"Features after multicollinearity removal: {n_features_after_multicol}")
    print(f"Removed {n_features_before_multicol - n_features_after_multicol} multicollinear features")
    return X, feature_columns

def perform_generalized_regression_analysis(df: pd.DataFrame, all_models: List[str], absolute_values: bool = False, drop_entailment: bool = False, standardize: bool = True) -> Dict[str, Any]:
    """
    Perform generalized linear regression analysis with interaction terms and polynomial features.
    
    This function extends the basic regression by adding:
    - Pairwise interaction terms between all predictors
    - Square term for the "a -> b" predictor
    
    The model includes:
    - Original predictors: 'a -> b', 'char(a ~ b)', 'token(a ~ b)', 'context switch'
    - Interaction terms: all pairwise combinations (e.g., 'a -> b' × 'char(a ~ b)')
    - Polynomial term: (a -> b)²
    
    Example usage:
        # Basic usage with all features
        generalized_stats = perform_generalized_regression_analysis(df, model_list)
        
        # With absolute values for entailment features
        generalized_stats = perform_generalized_regression_analysis(df, model_list, absolute_values=True)
        
        # Excluding entailment features
        generalized_stats = perform_generalized_regression_analysis(df, model_list, drop_entailment=True)
        
        # Without standardization
        generalized_stats = perform_generalized_regression_analysis(df, model_list, standardize=False)
    
    Args:
        df: DataFrame containing the data
        all_models: List of model names to analyze
        absolute_values: Should we convert all features to absolute values?
        drop_entailment: Should we ignore semantic entailment features?
        standardize: Should we standardize (normalize) features to mean=0, std=1? Default True
        all_models: List of model names to analyze
        absolute_values: Should we convert all features to absolute values?
        drop_entailment: Should we ignore semantic entailment features?
    
    Returns:
        Dictionary containing regression statistics for each model, including:
        - R2, adj_R2: Model fit statistics
        - f_stat, f_pvalue: Overall model significance
        - coefficients: Dictionary of feature coefficients
        - coef_pvalues: P-values for each coefficient
        - feature_columns: List of all features used in the model
        - n_features: Number of features in the expanded model
    """
    if drop_entailment:
        # See what happens if we drop the entailment similarity measure ('a -> b')
        base_feature_columns = global_feature_columns[1:]
    else:
        base_feature_columns = global_feature_columns

    regression_stats = {}  # Store regression statistics for each model

    # Prepare base features
    X_base = df[base_feature_columns].values
    if absolute_values:
        # Convert all features to absolute values. Will only affect entailment similarity 
        # measures ('a -> b', 'b -> a') since the remaining features are never negative.
        X_base = np.abs(X_base)

    # Add polynomial features and interaction terms and remove multicollinear features
    X, feature_columns = expand_feature_set(X_base, base_feature_columns, threshold=0.95)

    for model in all_models:
        y = df[model].values 
        
        # Standardize features if requested (default: True)
        if standardize:
            X = StandardScaler().fit_transform(X)
        
        # Create and fit the linear regression model
        reg_model = LinearRegression()
        reg_model.fit(X, y)
        
        # Make predictions for all rows (including those used for training)
        predictions = np.full(len(df), np.nan)
        predictions = reg_model.predict(X)
        
        # Calculate statistical significance
        n = len(y)  # number of observations
        p = len(feature_columns)  # number of features
        
        # Calculate residuals and standard errors
        y_pred = reg_model.predict(X)
        residuals = y - y_pred
        normalization_factor = n - p - 1 if n - p - 1 > 0 else 1
        mse = np.sum(residuals**2) / normalization_factor  # Mean squared error with degrees of freedom correction
        
        # Calculate R-squared and adjusted R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2_score = 1 - (ss_res / ss_tot)
        adj_r2 = 1 - (1 - r2_score) * (n - 1) / normalization_factor
        
        # F-statistic for overall model significance
        f_stat = (r2_score / p) / ((1 - r2_score) / normalization_factor)
        f_pvalue = 1 - stats.f.cdf(f_stat, p, normalization_factor)
        
        # Calculate standard errors for coefficients
        # X^T * X inverse for coefficient standard errors
        X_with_intercept = np.column_stack([np.ones(n), X])  # Add intercept column
        xtx_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        se_coef = np.sqrt(np.diagonal(xtx_inv) * mse)
        
        # T-statistics and p-values for each coefficient
        coef_with_intercept = np.concatenate([[reg_model.intercept_], reg_model.coef_])
        t_stats = coef_with_intercept / se_coef
        # Two-tailed p-values for coefficients
        coef_pvalues = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
        
        # Calculate relative contributions for standardized coefficients
        if standardize:
            abs_coefs = np.abs(reg_model.coef_)
            total_abs_coef = np.sum(abs_coefs)
            if total_abs_coef > 0:
                relative_contributions = {
                    feature: (abs(coef) / total_abs_coef) * 100 
                    for feature, coef in zip(feature_columns, reg_model.coef_)
                }
        
        # Store regression statistics
        regression_stats[model] = {
            'R2': r2_score,
            'adj_R2': adj_r2,
            'f_stat': f_stat,
            'f_pvalue': f_pvalue,
            'intercept': reg_model.intercept_,
            'intercept_pvalue': coef_pvalues[0],
            'coefficients': dict(zip(feature_columns, reg_model.coef_)),
            'coef_pvalues': dict(zip(feature_columns, coef_pvalues[1:])),
            'coef_se': dict(zip(['intercept'] + feature_columns, se_coef)),
            'relative_contributions': relative_contributions if standardize else {},
            'n': n,
            'n_features': p,
            'feature_columns': feature_columns,
            'standardized': standardize
        }
        
        # Print detailed statistics
        print(f"\nGeneralized Model: {model}")
        if standardize:
            print(f"  (Using standardized features)")
        print(f"  Features: {p} (including {len(base_feature_columns)} base + interactions + polynomial terms)")
        print(f"  R² = {r2_score:.4f}, Adjusted R² = {adj_r2:.4f}")
        print(f"  F-statistic = {f_stat:.4f}, F p-value = {f_pvalue:.6f}")
        print(f"  Intercept = {reg_model.intercept_:.4f} (p = {coef_pvalues[0]:.6f})")
        print("  Coefficients:")
        
        # Group and display coefficients by type for better readability
        base_coefs = []
        interaction_coefs = []
        poly_coefs = []
        
        for i, feature in enumerate(feature_columns):
            coef = reg_model.coef_[i]
            pval = coef_pvalues[i+1]
            significance = get_sig(pval)
            
            if feature in base_feature_columns:
                base_coefs.append((feature, coef, pval, significance))
            elif '×' in feature:
                interaction_coefs.append((feature, coef, pval, significance))
            elif '²' in feature:
                poly_coefs.append((feature, coef, pval, significance))
            else:
                base_coefs.append((feature, coef, pval, significance))
        
        if base_coefs:
            print("    Base features:")
            for feature, coef, pval, significance in base_coefs:
                if standardize and feature in relative_contributions:
                    rel_contrib = relative_contributions[feature]
                    print(f"      {feature}: {coef:.4f} (p = {pval:.6f}) {significance} [{rel_contrib:.1f}%]")
                else:
                    print(f"      {feature}: {coef:.4f} (p = {pval:.6f}) {significance}")
        
        if interaction_coefs:
            print("    Interaction terms:")
            for feature, coef, pval, significance in interaction_coefs:
                if standardize and feature in relative_contributions:
                    rel_contrib = relative_contributions[feature]
                    print(f"      {feature}: {coef:.4f} (p = {pval:.6f}) {significance} [{rel_contrib:.1f}%]")
                else:
                    print(f"      {feature}: {coef:.4f} (p = {pval:.6f}) {significance}")
        
        if poly_coefs:
            print("    Polynomial terms:")
            for feature, coef, pval, significance in poly_coefs:
                if standardize and feature in relative_contributions:
                    rel_contrib = relative_contributions[feature]
                    print(f"      {feature}: {coef:.4f} (p = {pval:.6f}) {significance} [{rel_contrib:.1f}%]")
                else:
                    print(f"      {feature}: {coef:.4f} (p = {pval:.6f}) {significance}")

    return regression_stats


def remove_multicollinear_features(X: np.ndarray, feature_names: List[str], threshold: float = 0.95) -> tuple:
    """
    Remove features that are highly correlated with other features to avoid multicollinearity.
    
    Args:
        X: Feature matrix
        feature_names: List of feature names
        threshold: Correlation threshold above which features are considered multicollinear
    
    Returns:
        Tuple of (cleaned_X, cleaned_feature_names)
    """
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X.T)
    
    # Find pairs of highly correlated features
    to_remove = set()
    n_features = X.shape[1]
    
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if abs(corr_matrix[i, j]) > threshold:
                # Remove the feature with lower variance (less informative)
                var_i = np.var(X[:, i])
                var_j = np.var(X[:, j])
                if var_i < var_j:
                    to_remove.add(i)
                else:
                    to_remove.add(j)
    
    # Keep features that are not marked for removal
    keep_indices = [i for i in range(n_features) if i not in to_remove]
    
    if len(to_remove) > 0:
        removed_features = [feature_names[i] for i in to_remove]
        print(f"Removed {len(to_remove)} multicollinear features: {removed_features}")
    
    X_clean = X[:, keep_indices]
    clean_feature_names = [feature_names[i] for i in keep_indices]
    
    return X_clean, clean_feature_names


def perform_lasso_analysis(
    df: pd.DataFrame,
    all_models: List[str],
    n_iterations: int = 10,
    sample_fraction: float = 0.8,
    random_state: int | None = None,
    alphas: List[float] | None = None,
    standardize: bool = True,
) -> Dict[str, Any]:
    """Run repeated LASSO (L1) regression to assess relative predictor contributions.

    For each model we:
      * Build feature matrix with predictors: abs('a -> b'), 'char(a ~ b)', 'token(a ~ b)', 'context switch'
      * Repeat n_iterations:
          - Randomly sample (without replacement) a fraction of rows (sample_fraction)
          - Standardize features if standardize=True (default)
          - Fit LassoCV (if multiple alphas) or single-alpha Lasso via LassoCV using provided alphas or default
          - Record coefficients (excluding intercept)
      * Aggregate:
          - mean_coef (signed)
          - mean_abs_coef
          - selection_freq (proportion of iterations with non-zero coefficient)
          - relative_mean_abs_weight (mean_abs_coef / sum(mean_abs_coef))

    Args:
        df: DataFrame containing the data
        all_models: List of model names to analyze
        n_iterations: Number of bootstrap iterations (default: 10)
        sample_fraction: Fraction of data to sample in each iteration (default: 0.8)
        random_state: Random seed for reproducibility
        alphas: Alpha values for LASSO regularization (default: logspace(-3, 1, 30))
        standardize: Should we standardize (normalize) features to mean=0, std=1? Default True

    Returns dict keyed by model -> {
        'iterations': n_effective_iterations,
        'feature_stats': { feature: {...stats...} },
        'alpha_distribution': list of best alpha per iteration,
        'standardized': bool indicating if features were standardized
    }
    """
    rng = np.random.default_rng(random_state)
    feature_columns = ['a -> b', 'char(a ~ b)', 'token(a ~ b)', 'context switch']
    # Prepare absolute entailment feature
    # Build matrix once (will resample rows later)
    feat_df = df[feature_columns].copy()
    if feat_df.empty:
        return {}
    feat_df['a -> b'] = feat_df['a -> b'].abs()
    # Rename for clarity in output
    renamed_features = ['abs(a -> b)', 'char(a ~ b)', 'token(a ~ b)', 'context switch']

    lasso_results: Dict[str, Any] = {}

    # Default alpha grid if not provided
    if alphas is None:
        alphas = np.logspace(-3, 1, 30)

    for model in all_models:
        if model not in df.columns:
            continue
        y_full = df[model].values
        X_full = feat_df.values.astype(float)

        # Filter out rows with NaN in either X or y
        valid_mask = ~(np.isnan(X_full).any(axis=1) | np.isnan(y_full))
        X_full_valid = X_full[valid_mask]
        y_full_valid = y_full[valid_mask]
        n_rows = X_full_valid.shape[0]
        if n_rows < 10:  # heuristic threshold
            continue

        coef_records = []  # list of coefficient arrays per iteration
        alpha_list = []
        eff_iterations = 0
        sample_size = max(4, int(sample_fraction * n_rows))

        for it in range(n_iterations):
            if sample_size >= n_rows:
                sample_idx = np.arange(n_rows)
            else:
                sample_idx = rng.choice(n_rows, size=sample_size, replace=False)
            X_sample = X_full_valid[sample_idx]
            y_sample = y_full_valid[sample_idx]

            # Guard against degenerate sample
            if np.std(y_sample) == 0:
                continue

            # Standardize features if requested (default: True)
            if standardize:
                X_sample = StandardScaler().fit_transform(X_sample)
            
            lasso_cv = LassoCV(alphas=alphas, cv=min(5, max(2, sample_size // 4)), random_state=rng.integers(0, 1_000_000))
            lasso_cv.fit(X_sample, y_sample)
            coefs = lasso_cv.coef_
            coef_records.append(coefs)
            alpha_list.append(lasso_cv.alpha_)
            eff_iterations += 1

        if eff_iterations == 0:
            continue

        coef_matrix = np.vstack(coef_records)  # shape (eff_iterations, n_features)
        mean_coef = coef_matrix.mean(axis=0)
        abs_coef = np.abs(coef_matrix)
        mean_abs_coef = abs_coef.mean(axis=0)
        selection_freq = (coef_matrix != 0).sum(axis=0) / eff_iterations
        total_mean_abs = mean_abs_coef.sum()
        if total_mean_abs == 0:
            relative_mean_abs = np.zeros_like(mean_abs_coef)
        else:
            relative_mean_abs = mean_abs_coef / total_mean_abs

        feature_stats = {}
        for fname, mc, mac, freq, rel in zip(renamed_features, mean_coef, mean_abs_coef, selection_freq, relative_mean_abs):
            feature_stats[fname] = {
                'mean_coef': mc,
                'mean_abs_coef': mac,
                'selection_freq': freq,
                'relative_mean_abs_weight': rel,
            }

        lasso_results[model] = {
            'iterations': eff_iterations,
            'feature_stats': feature_stats,
            'alpha_distribution': alpha_list,
            'alpha_mean': float(np.mean(alpha_list)) if alpha_list else None,
            'alpha_std': float(np.std(alpha_list)) if alpha_list else None,
            'standardized': standardize,
        }

        # Console summary for quick inspection
        print(f"\nLASSO summary for model: {model} (iterations={eff_iterations})")
        if standardize:
            print(f"  (Using standardized features)")
        for fname, stats_d in feature_stats.items():
            rel_weight_pct = stats_d['relative_mean_abs_weight'] * 100
            if standardize:
                print(
                    f"  {fname:<16} sel_freq={stats_d['selection_freq']:.2f} "
                    f"mean_abs={stats_d['mean_abs_coef']:.4f} rel_weight={stats_d['relative_mean_abs_weight']:.2f} "
                    f"[{rel_weight_pct:.1f}%]"
                )
            else:
                print(
                    f"  {fname:<16} sel_freq={stats_d['selection_freq']:.2f} "
                    f"mean_abs={stats_d['mean_abs_coef']:.4f} rel_weight={stats_d['relative_mean_abs_weight']:.2f}"
                )

    return lasso_results

def create_lasso_summary(lasso_stats: Dict[str, Any]) -> pd.DataFrame:
    """Flatten lasso_stats into a tabular DataFrame.

    Columns: model, feature, mean_coef, mean_abs_coef, selection_freq, relative_mean_abs_weight, iterations, alpha_mean, alpha_std
    """
    rows = []
    model_list = list(lasso_stats.keys())
    
    for i, (model, model_stats) in enumerate(lasso_stats.items()):
        iters = model_stats.get('iterations')
        alpha_mean = model_stats.get('alpha_mean')
        alpha_std = model_stats.get('alpha_std')
        for feature, fstats in model_stats['feature_stats'].items():
            # Convert relative_mean_abs_weight to percentage for display
            contribution_pct = fstats['relative_mean_abs_weight'] * 100
            rows.append({
                'model': model,
                'feature': feature,
                'mean_coef': fstats['mean_coef'],
                'mean_abs_coef': fstats['mean_abs_coef'],
                'selection_freq': fstats['selection_freq'],
                'relative_mean_abs_weight': fstats['relative_mean_abs_weight'],
                'contribution_pct': contribution_pct,
                'iterations': iters,
                'alpha_mean': alpha_mean,
                'alpha_std': alpha_std,
            })
        
        # Add empty row between models (except after the last model)
        if i < len(model_list) - 1:
            rows.append({
                'model': None,
                'feature': None,
                'mean_coef': None,
                'mean_abs_coef': None,
                'selection_freq': None,
                'relative_mean_abs_weight': None,
                'contribution_pct': None,
                'iterations': None,
                'alpha_mean': None,
                'alpha_std': None,
            })
    
    return pd.DataFrame(rows)

def get_entailment_ranks():
    """
      Add semantic entailment ranking to the each model.
    """
    global models
    semantic_ranking_df = pd.read_excel('train_results_summary.xlsx', sheet_name='Semantic Ranking')
    ranked_models = semantic_ranking_df['model'].tolist()
    for rank, model in enumerate(ranked_models):
        models[model]['entailment_rank'] = rank + 1

def create_entailment_contribution_worksheet(gen_regression_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a worksheet analyzing the 'a -> b' contribution from generalized regression results.
    
    Models are sorted by descending 'Contribution (%)' of the 'coef_a -> b' metric.
    Includes MTEB rankings and semantic entailment rankings, and calculates Spearman correlations 
    between contribution rankings and both MTEB rankings and semantic rankings.
    
    Args:
        gen_regression_results: Results from perform_generalized_regression_analysis
        
    Returns:
        DataFrame with models sorted by 'a -> b' contribution and correlation analysis
    """
    from scipy.stats import spearmanr

    get_entailment_ranks()
    # Extract 'a -> b' contributions for each model
    model_contributions = []
    
    for model, stats in gen_regression_results.items():
        # Look for 'a -> b' in relative_contributions
        relative_contributions = stats.get('relative_contributions', {})
        
        # Find the 'a -> b' contribution (could be exact match or part of a more complex feature name)
        entailment_contribution = None
        for feature, contribution in relative_contributions.items():
            if feature == 'a -> b':
                entailment_contribution = contribution
                break
        
        if entailment_contribution is not None:
            model_contributions.append({
                'model': model,
                'entailment_contribution_pct': entailment_contribution,
                'mteb_rank': models[model]['mteb_rank'],
                'entailment_rank': models[model]['entailment_rank']
            })
    
    # Sort by descending entailment contribution
    model_contributions.sort(key=lambda x: x['entailment_contribution_pct'], reverse=True)
    
    # Create contribution rankings (1 = highest contribution)
    for i, item in enumerate(model_contributions):
        item['contribution_rank'] = i + 1
    
    # Filter out models for correlation analysis
    models_with_mteb = [item for item in model_contributions if item['mteb_rank'] is not None]
    models_with_entailment = [item for item in model_contributions if item['entailment_rank'] is not None]
    
    # Calculate Spearman correlations if we have enough data
    mteb_correlation_result = None
    semantic_correlation_result = None
    
    contribution_ranks = [item['contribution_rank'] for item in models_with_mteb]
    mteb_ranks = [item['mteb_rank'] for item in models_with_mteb]
    
    correlation, p_value = spearmanr(contribution_ranks, mteb_ranks)
    mteb_correlation_result = {
        'correlation': correlation,
        'p_value': p_value,
        'n_models': len(models_with_mteb)
    }
    
    contribution_ranks = [item['contribution_rank'] for item in models_with_entailment]
    entailment_ranks = [item['entailment_rank'] for item in models_with_entailment]
    
    correlation, p_value = spearmanr(contribution_ranks, entailment_ranks)
    semantic_correlation_result = {
        'correlation': correlation,
        'p_value': p_value,
        'n_models': len(models_with_entailment)
    }
    
    # Create DataFrame rows
    rows = []
    
    # Add model data
    for item in model_contributions:
        rows.append({
            'model': item['model'],
            'entailment_contribution_pct': f"{item['entailment_contribution_pct']:.2f}",
            'contribution_rank': item['contribution_rank'],
            'mteb_rank': item['mteb_rank'] if item['mteb_rank'] is not None else 'N/A',
            'entailment_rank': item['entailment_rank'] if item['entailment_rank'] is not None else 'N/A'
        })
    
    # Add empty row before correlation results
    rows.append({
        'model': '',
        'entailment_contribution_pct': '',
        'contribution_rank': '',
        'mteb_rank': '',
        'entailment_rank': ''
    })
    
    # Add correlation results
    if mteb_correlation_result:
        rows.append({
            'model': 'MTEB Correlation Analysis:',
            'entailment_contribution_pct': '',
            'contribution_rank': '',
            'mteb_rank': '',
            'entailment_rank': ''
        })
        rows.append({
            'model': f"Correlation coefficient: {mteb_correlation_result['correlation']:.4f}",
            'entailment_contribution_pct': '',
            'contribution_rank': '',
            'mteb_rank': '',
            'entailment_rank': ''
        })
        rows.append({
            'model': f"P-value: {mteb_correlation_result['p_value']:.6f}",
            'entailment_contribution_pct': '',
            'contribution_rank': '',
            'mteb_rank': '',
            'entailment_rank': ''
        })
        rows.append({
            'model': f"Number of models: {mteb_correlation_result['n_models']}",
            'entailment_contribution_pct': '',
            'contribution_rank': '',
            'mteb_rank': '',
            'entailment_rank': ''
        })
        
        # Add empty row between correlations
        rows.append({
            'model': '',
            'entailment_contribution_pct': '',
            'contribution_rank': '',
            'mteb_rank': '',
            'entailment_rank': ''
        })
    else:
        rows.append({
            'model': 'MTEB Correlation: Insufficient data',
            'entailment_contribution_pct': '(Need at least 3 models with MTEB ranks)',
            'contribution_rank': '',
            'mteb_rank': '',
            'entailment_rank': ''
        })
    
    if semantic_correlation_result:
        rows.append({
            'model': 'Semantic Ranking Correlation Analysis:',
            'entailment_contribution_pct': '',
            'contribution_rank': '',
            'mteb_rank': '',
            'entailment_rank': ''
        })
        rows.append({
            'model': f"Correlation coefficient: {semantic_correlation_result['correlation']:.4f}",
            'entailment_contribution_pct': '',
            'contribution_rank': '',
            'mteb_rank': '',
            'entailment_rank': ''
        })
        rows.append({
            'model': f"P-value: {semantic_correlation_result['p_value']:.6f}",
            'entailment_contribution_pct': '',
            'contribution_rank': '',
            'mteb_rank': '',
            'entailment_rank': ''
        })
        rows.append({
            'model': f"Number of models: {semantic_correlation_result['n_models']}",
            'entailment_contribution_pct': '',
            'contribution_rank': '',
            'mteb_rank': '',
            'entailment_rank': ''
        })
    else:
        rows.append({
            'model': 'Semantic Ranking Correlation: Insufficient data',
            'entailment_contribution_pct': '(Need at least 3 models with semantic ranks)',
            'contribution_rank': '',
            'mteb_rank': '',
            'entailment_rank': ''
        })
    
    return pd.DataFrame(rows)


def create_worksheet(validation_results: List[Dict], dataset: List[Dict], data_type: str) -> pd.DataFrame:
    """
    Create worksheet for similarity estimates.
    
    Args:
        validation_results: List of validation result dictionaries
        dataset: List of dataset records
        data_type: Either 'baseline' or 'projected' to specify which data to extract
    
    Returns:
        DataFrame with similarity estimates organized by dataset records and models
    """
    rows = []
    
    # Group validation results by endpoint and out_dim for easier access
    validation_dict = {}
    for result in validation_results:
        key = (result['endpoint'], result['out_dim'])
        if key not in validation_dict:
            validation_dict[key] = {}
        validation_dict[key][result['model']] = result.get(data_type, [])
    
    # Get all unique models for column headers
    all_models = sorted(models.keys())
    
    # Get all unique endpoints and out_dims
    endpoints = sorted(set(result['endpoint'] for result in validation_results))
    out_dims = sorted(set(result['out_dim'] for result in validation_results))
    
    # Create rows for each dataset record and endpoint/out_dim combination
    for endpoint in endpoints:
        for out_dim in out_dims:
            key = (endpoint, out_dim)
            if key not in validation_dict:
                continue
                
            for idx, record in enumerate(dataset):
                row = {key: record[key] for key in ['a', 'b', endpoint]}
                if data_type == 'baseline':
                    row.update({key: record[key] for key in global_feature_columns if not key in row})
                row['trained endpoint'] = endpoint
                row['out_dim'] = out_dim
                
                # Add model predictions as columns
                for model in all_models:
                    row[model] = validation_dict[key][model][idx]
                
                rows.append(row)
    
    # After collecting all rows, create regression estimators for the cosine similarities 
    # produced by various embedding models
    df = pd.DataFrame(rows)
    
    if data_type == 'baseline':
        # Regression analysis with all factors (standardized by default)
        df.attrs['Lin. Regression'] = perform_regression_analysis(df, all_models)
        # Regression analysis with semantic entailment features converted to absolute values
        df.attrs['Lin. Regression (abs)'] = perform_regression_analysis(df, all_models, absolute_values=True)
        # Regression analysis without semantic entailment features
        df.attrs['Lin. Regression (wo entailment)'] = perform_regression_analysis(df, all_models, drop_entailment=True)
        # Generalized regression analysis with interaction terms and polynomial features
        df.attrs['Gen. Regression'] = perform_generalized_regression_analysis(df, all_models)

        # Create a separate worksheet for 'a -> b' contribution analysis
        df.attrs['Gen. Reg. a->b contribution'] = create_entailment_contribution_worksheet(df.attrs['Gen. Regression'])

        df.attrs['Gen. Regression (wo entailment)'] = perform_generalized_regression_analysis(df, all_models, drop_entailment=True)
        # LASSO feature contribution analysis
        df.attrs['LASSO'] = perform_lasso_analysis(df, all_models)
    
    return df.sort_values(by='out_dim')

def get_row(model:str|None=None, metric:str|None=None, value:float|None=None, p_value:float|None=None, significance:str|None=None, n_obs:int|None=None, contribution_pct:float|None=None) -> dict:
    return {
        'model': model,
        'metric': metric,
        'value': value,
        'p_value': p_value,
        'q_value': None,
        'significance': significance,
        'contribution_pct': contribution_pct,
        'n': n_obs
    }

def create_regression_summary(regression_stats: dict) -> pd.DataFrame:
    """
    Create a DataFrame with regression analysis results in columnar format.
    
    Args:
        regression_stats: Dictionary containing regression statistics for each model
    
    Returns:
        DataFrame with regression analysis summary
    """
    summary_rows = []
    
    for model, stats in regression_stats.items():
        if 'error' in stats:
            # Handle models where regression failed
            summary_rows.append(get_row(model, 'Error', stats['error'], n_obs=stats['n']))
        else:
            # overall statistics
            summary_rows.extend([
                get_row(model, 'R²', stats['R2']),
                get_row(model, 'Adjusted R²', stats['adj_R2']),
                get_row(model, 'F-statistic', stats['f_stat'], stats['f_pvalue'], n_obs=stats['n']),
            ])
            
            summary_rows.append(get_row(model, 'Intercept', stats['intercept'], stats['intercept_pvalue'], get_sig(stats['intercept_pvalue'])))
            
            # Get relative contributions if available
            relative_contributions = stats.get('relative_contributions', {})
            
            for feature, coef_value in stats['coefficients'].items():
                pvalue = stats['coef_pvalues'][feature]
                contribution_pct = relative_contributions.get(feature, None)
                summary_rows.append(get_row(model, f'coef_{feature}', coef_value, pvalue, contribution_pct=contribution_pct))
            
            # add empty row for separation
            summary_rows.append(get_row())
    
    # Calculate FDR-BY q-values for all numeric p_value entries
    df = pd.DataFrame(summary_rows)

    # Benjamini-Yekutieli correction (controls FDR under dependency)
    mask = df['p_value'].notna()
    pvals = df['p_value'][mask]
    
    if len(pvals) > 0:  # Only apply correction if there are p-values to correct
        _, qvals, _, _ = multipletests(pvals.values, method='fdr_by')
        df.loc[mask, 'q_value'] = qvals
        df.loc[mask, 'significance'] = df['q_value'][mask].apply(get_sig)
    else:
        # If no p-values, set significance based on original p-values where available
        df['significance'] = df['p_value'].apply(lambda x: get_sig(x) if pd.notna(x) else None) 

    return df

def get_index(df: pd.DataFrame, metric: str="Adjusted R²") -> pd.Series:
    return df['metric'] == metric

def create_r2_summary(regression_summary_df: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a summary DataFrame showing R² differences between full model and variants.
    
    Args:
        regression_summary_df: Dictionary containing regression summaries with keys:
                              'Regression', 'Regression (abs)', 'Regression (wo entailment)'
    
    Returns:
        DataFrame with R² values and differences, including mean and std dev rows
    """
    if not regression_summary_df:
        return None
    
    bl = regression_summary_df['Lin. Regression']
    abs_reg = regression_summary_df['Lin. Regression (abs)']
    wo_reg = regression_summary_df['Lin. Regression (wo entailment)']

    # NB! by default, get_index will return indices for 'Adjusted R²'
    r2_diff = pd.DataFrame({
        'model': bl['model'][get_index(bl)].values,
        'R2 (baseline)': pd.to_numeric(bl['value'][get_index(bl)].values),
        'R2 (abs)': pd.to_numeric(abs_reg['value'][get_index(abs_reg)].values),
        'R2 (wo entailment)': pd.to_numeric(wo_reg['value'][get_index(wo_reg)].values),
    })
    
    # Calculate differences
    r2_diff['R2 - R2(abs)'] = r2_diff['R2 (baseline)'] - r2_diff['R2 (abs)']
    r2_diff['R2(abs) - R2(wo entailment)'] = r2_diff['R2 (abs)'] - r2_diff['R2 (wo entailment)']
    
    # Add summary statistics
    numeric_cols = r2_diff.columns[1:]  # Exclude 'Model' column
    
    # Add mean row
    mean_row = r2_diff[numeric_cols].mean()
    mean_row['model'] = 'mean'
    r2_diff.loc['mean'] = mean_row
    
    # Add standard deviation row
    std_row = r2_diff[numeric_cols].std()
    std_row['model'] = 'std dev'
    r2_diff.loc['std dev'] = std_row

    return r2_diff.reset_index(drop=True)

def create_generalized_r2_summary(regression_summary_df: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a summary DataFrame showing R² and Adjusted R² differences between 
    Generalized Regression and Gen. Regression (wo entailment).
    
    Args:
        regression_summary_df: Dictionary containing regression summaries with keys:
                              'Generalized Regression', 'Gen. Regression (wo entailment)'
    
    Returns:
        DataFrame with R² values and differences, including mean and std dev rows
    """
    gen_reg = regression_summary_df['Gen. Regression']
    gen_reg_wo = regression_summary_df['Gen. Regression (wo entailment)']

    # Reporting only adjusted R² values in Excel
    gen_adj_r2_idx = get_index(gen_reg, 'Adjusted R²')
    gen_wo_adj_r2_idx = get_index(gen_reg_wo, 'Adjusted R²')

    gen_r2_df = pd.DataFrame({
        'model': gen_reg['model'][gen_adj_r2_idx].values,
        'Gen. Regression R²': pd.to_numeric(gen_reg['value'][gen_adj_r2_idx].values),
        'Gen. Reg. (wo entailment) R²': pd.to_numeric(gen_reg_wo['value'][gen_wo_adj_r2_idx].values),
    })
    
    # Calculate differences
    gen_r2_df['R² Difference'] = gen_r2_df['Gen. Regression R²'] - gen_r2_df['Gen. Reg. (wo entailment) R²']
    
    # Add summary statistics
    numeric_cols = gen_r2_df.columns[1:]  # Exclude 'model' column
    
    # Add mean row
    mean_row = gen_r2_df[numeric_cols].mean()
    mean_row['model'] = 'mean'
    gen_r2_df.loc['mean'] = mean_row
    
    # Add standard deviation row
    std_row = gen_r2_df[numeric_cols].std()
    std_row['model'] = 'std dev'
    gen_r2_df.loc['std dev'] = std_row
    
    return gen_r2_df.reset_index(drop=True)

def create_r2_paired_tests(r2_diff: pd.DataFrame) -> pd.DataFrame:
    """Create a DataFrame with paired t-test results between R² variants.

    Performs paired (within-model) t-tests comparing:
      1. Full Model vs Abs Values
      2. Full Model vs No Entailment
      3. Abs Values vs No Entailment

    Excludes the summary rows (Mean, Std Dev) if present. Only models with
    non-missing values for a given pair are included in that pair's test.

    Returns:
        DataFrame with columns:
            Comparison, N, Mean_1, Mean_2, Mean_Diff, Std_Diff,
            t_stat, p_value, q_value (BY FDR), Significance
    """
    # Work on a copy without summary rows
    df = r2_diff.copy()
    df = df[~df['model'].isin(['mean', 'std dev'])]

    comparisons = [
        ('R2 (baseline)', 'R2 (abs)'),
        ('R2 (baseline)', 'R2 (wo entailment)'),
        ('R2 (abs)', 'R2 (wo entailment)'),
    ]

    rows = []
    for col1, col2 in comparisons:
        sub = df[[col1, col2]].dropna()
        x = sub[col1].astype(float)
        y = sub[col2].astype(float)
        diff = x - y
        mean_diff = diff.mean()
        std_diff = diff.std(ddof=1)

        # paired t-test
        t_stat, p_val = stats.ttest_rel(x, y, nan_policy='omit')
        rows.append({
            'comparison': f'{col1} vs {col2}',
            'n': len(x),
            'mean_1': x.mean(),
            'mean_2': y.mean(),
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            't_stat': t_stat,
            'p_value': p_val,
            'q_value': None,
            'significance': None,
        })

    data = pd.DataFrame(rows)

    # Multiple testing correction (Benjamini-Yekutieli) on available p-values
    _, qvals, _, _ = multipletests(data['p_value'], method='fdr_by')
    data['q_value'] = qvals
    data['significance'] = data['q_value'].apply(get_sig)
    return data

def get_sig(p_val: float) -> str:
    return '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''

def main(validation_file: str, dataset_file: str = None, output_file: str = None):
    """
    Main function to process validation results and create Excel output.
    """
    # Load validation results
    validation_results = load_validation_results(validation_file)
    print(f"Loaded {len(validation_results)} validation results from '{validation_file}'")
    
    # Determine dataset file if not provided
    if dataset_file is None:
        # Extract dataset name from validation filename
        # e.g., 'validation_results_entailment_minitest.json' -> 'entailment_minitest.json'
        basename = os.path.basename(validation_file)
        if basename.startswith('validation_results_'):
            dataset_name = basename.replace('validation_results_', '')
            dataset_file = os.path.join(os.path.dirname(validation_file), dataset_name)
        else:
            raise ValueError(f"Cannot determine dataset file from validation file '{validation_file}'. Please specify --dataset.")
    
    # Load dataset
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file '{dataset_file}' not found.")
    
    dataset = load_dataset(dataset_file)
    print(f"Loaded {len(dataset)} dataset records from '{dataset_file}'")
    
    # Create worksheets
    print("Creating baseline worksheet...")
    baseline_df = create_worksheet(validation_results, dataset, 'baseline')
    print("✓ Baseline worksheet created with regression analyses:")
    print("  - Standard regression")
    print("  - Regression with absolute values") 
    print("  - Regression without entailment features")
    print("  - Generalized regression (with interactions + polynomial)")
    print("  - LASSO feature analysis")

    print("Creating projected worksheet...")
    projected_df = create_worksheet(validation_results, dataset, 'projected')
    
    # Determine output file if not provided
    if output_file is None:
        output_file = os.path.basename(validation_file).replace('.json', '_summary.xlsx')

    print("Creating regression analysis summary...")
    regression_summary_df = {}
    for key in baseline_df.attrs:
        if key == 'LASSO':
            lasso_df = create_lasso_summary(baseline_df.attrs[key])
            # Rename columns for better display in Excel
            lasso_df = lasso_df.rename(columns={
                'contribution_pct': 'Contribution (%)',
                'relative_mean_abs_weight': 'Relative Weight'
            })
            regression_summary_df['LASSO'] = lasso_df
        elif isinstance(baseline_df.attrs[key], pd.DataFrame):
            # Skip DataFrame objects (e.g., 'Gen. Reg. a->b contribution')
            # These are already processed DataFrames, not raw regression statistics
            regression_summary_df[key] = baseline_df.attrs[key]
        else:
            reg_df = create_regression_summary(baseline_df.attrs[key])
            # Rename columns for better display in Excel
            reg_df = reg_df.rename(columns={
                'contribution_pct': 'Contribution (%)'
            })
            regression_summary_df[key] = reg_df

    # add worksheet displaying the difference between R2
    r2_diff = create_r2_summary(regression_summary_df)
    regression_summary_df['Lin. Reg. R2 differences'] = r2_diff

    # Add generalized regression R² differences
    gen_r2_diff = create_generalized_r2_summary(regression_summary_df)
    if gen_r2_diff is not None:
        regression_summary_df['Gen. Reg. R2 differences'] = gen_r2_diff

    # Add paired t-test significance analysis between R2 variants
    regression_summary_df['R2 paired tests'] = create_r2_paired_tests(r2_diff)

    sheets_created = ['Baseline', 'Projected']
    # Write to Excel with separate worksheets
    print(f"Writing results to '{output_file}'...")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        baseline_df.to_excel(writer, sheet_name='Baseline', index=False)
        projected_df.to_excel(writer, sheet_name='Projected', index=False)
        if regression_summary_df:
            for key, df in regression_summary_df.items():
                sheets_created.append(key)
                df.to_excel(writer, sheet_name=key, index=False)
    
    print(f"Successfully created Excel file with {len(baseline_df)} baseline rows and {len(projected_df)} projected rows.")
    print(f"Worksheets created: {', '.join(sheets_created)}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize validation results into Excel format.")

    parser.add_argument("--validation_file", type=str, default="out/cosine_similarity_trial.json",
                      help="Path to the validation results JSON file (default: 'out/cosine_similarity_trial.json')")
    parser.add_argument("--dataset", type=str, default="annotations/trial.json",
                      help="Path to the corresponding dataset JSON file (default: 'annotations/trial.json').")
    parser.add_argument("--output", type=str, default=None,
                      help="Path to the output Excel file. If not provided, will be generated based on input filename.")
    
    args = parser.parse_args()
    main(args.validation_file, args.dataset, args.output)
