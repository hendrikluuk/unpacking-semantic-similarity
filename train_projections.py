#!/usr/bin/env python3
"""
  Train a projection model on a dataset of embeddings.
"""
import os
import json
import argparse

from ml.trainer import Trainer
from ml.config import configurations

endpoints = ['a -> b', 'b -> a', 'char(a ~ b)', 'token(a ~ b)']
model_folder = 'models'

def main(out_dim: int = 0, save_model: bool = False, overwrite: bool = False):
    results = []

    if save_model:
        # make sure the model directory exists
        os.makedirs(model_folder, exist_ok=True)

    existing_results = {}
    if not overwrite:
        result_file = f'out/train_results{out_dim}.json'
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results = json.load(f)
            existing_results = {f"{r['model']}/{r['dataset']}/{r['out_dim']}": True for r in results}
            print(f"Found {len(results)} existing results for out_dim={out_dim}, will add only missing results.")

    for conf_key in configurations:
        dataset, model = conf_key.split('/')
        print(f"Started training the projection of embeddings from '{model}' on '{dataset}' dataset")

        if not overwrite and f"{model}/{dataset}/{out_dim}" in existing_results:
            print(f"Skipping {model}/{dataset} as it already exists in results.")
            continue

        for endpoint in endpoints:
            print(f"Training endpoint: '{endpoint}'")

            # run with default model configuration if out_dim is 0
            model_config = {} if out_dim == 0 else {'out_dim': out_dim} 

            t = Trainer(conf_key, data={'endpoint': endpoint}, model=model_config)
            r = t.train()

            out_dim = t.model.config.out_dim
            if save_model:
                # save the model into a file with input embedding model name, dataset name, endpoint and output dimension
                model_name = f"{model}_{dataset}_{endpoint.replace(' ', '')}_{out_dim}.json"
                t.save_model(f"{model_folder}/{model_name}")

            eval_result = t.model.evaluate_metrics(t.data['val']['a'], t.data['val']['b'], t.data['val']['y'])
            eval_result = {'model': model, 'dataset': dataset, 'endpoint': endpoint, 'out_dim': t.config['model']['out_dim'], **eval_result}
            results.append(eval_result)

    print("Training completed for all models and datasets.")

    outfile = f"out/train_results{out_dim}.json" if out_dim > 0 else f"out/train_results{t.config['model']['out_dim']}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train projection models on datasets.")
    parser.add_argument('--out_dim', type=int, default=0, help='Output dimension for the projection model. If 0, use default configuration.')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model to disk.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results instead of appending.')
    args = parser.parse_args()
    main(args.out_dim, args.save_model)