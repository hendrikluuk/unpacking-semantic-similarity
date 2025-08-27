#!/usr/bin/env python3
"""
Predict cosine similarity of sentence pairs in 'annotations/trial.json'
using the original (unprojected) embeddings.
"""
import os
import glob
import json
import pickle
import argparse

import numpy as np

from ml.similarity_learner import SimilarityLearner
from ml.config import configurations

# adapted from ml.Trainer
def load_data(dataset:str, model:str, verbose=True, include_original_data:bool=False) -> dict:
    source = f"{dataset}/{model}"
    data_file = None
    endpoint = "a -> b"

    if source in configurations:
        # "dataset" entry is a list
        data_file = configurations[source]["data"]["dataset"][0]
        endpoint = configurations[source]["data"]["endpoint"]
    else:
        # datasets that we are not training on e.g. 'annotations/trial.json'
        # will not have a corresponding entry in {configurations}
        file_path = os.path.join('cache', model, f"{dataset}.pkl")
        if os.path.isfile(file_path):
            data_file = file_path
    if not data_file:
        raise RuntimeError(f"File '{data_file}' was not found!")

    if verbose:
        print(f"Loading dataset from '{data_file}'")
    with open(data_file, "rb") as data_file:
        # load the data from the file
        # the file should contain a dictionary with keys 'x' and 'y'
        # where 'x' is a list of input sequences and 'y' is a list of target sequences
        # both are numpy arrays
        data = pickle.load(data_file)

    result = {}
    for key in ['a', 'b']:
        # stack embeddings by rows into a matrix
        result[key] = np.stack([_get_embedding(pair[key], data) for pair in data['data']], dtype=np.float32)

    result['y'] = np.array([pair[endpoint] for pair in data['data']], dtype=np.float32).reshape(-1, 1)
    assert result['a'].shape[0] == result['b'].shape[0] == result['y'].shape[0], \
        f"a, b or y have mismatched shapes: a {result['a'].shape}, b {result['b'].shape}, y {result['y'].shape}"

    print(f"Size of dataset: {len(result['y'])}")
    if include_original_data:
        result['data'] = data['data']
    return result

def get_models(pattern:str) -> list[str]:
    """
    Get a list of model names matching the pattern in the 'models' directory.
    """
    models = glob.glob(f"models/{pattern}")
    if not models:
        print(f"No models found matching pattern: {pattern}")
    return models

def main(model_pattern:str="*_propositions_a->b_*.json", dataset:str="trial", outfile:str=None):
    # Loop through all models in the 'models' directory
    models = get_models(model_pattern)
    if not models:
        return

    results = []
    for model_path in models:
        model_name, _, endpoint, outdim = os.path.basename(model_path).replace('.json', '').rsplit('_', 3)
        print(f"Evaluating model: {model_name}")

        endpoint = endpoint.replace('->', ' -> ').replace('~', ' ~ ')

        # Load the semantic projection for the embedding model and endpoint
        learner = SimilarityLearner().load(model_path)
        
        # Load the dataset encoded by the corresponding embedding model
        data = load_data(dataset, model_name)

        # Evaluate the model on data
        metrics = learner.predict(data['a'], data['b'], data['y'], include_baseline=True)

        results.append({
            'model': model_name,
            'dataset': dataset,
            'endpoint': endpoint,
            'out_dim': outdim,
            **metrics
        })

    if not outfile:
        outfile = f'out/cosine_similarity_{dataset}.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Validation results ({len(results)} items) saved to '{outfile}'")

# same as in ml.Trainer
def _get_embedding(key:str, data:dict) -> np.ndarray:
    key = key.replace('[', '').replace(']', '').strip()
    return data['embeddings'][data['index'][key]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate projections/models on a dataset.")
    parser.add_argument('--pattern', type=str, default="*_propositions_a->b_*.json",
                        help="Pattern to match model files in the 'models' directory.")
    parser.add_argument('--dataset', type=str, default="trial",
                        help="Dataset to validate the models on.")
    parser.add_argument('--outfile', type=str, default=None,
                        help="Output file to save the validation results.")
    args = parser.parse_args()
    main(args.pattern, args.dataset, args.outfile)