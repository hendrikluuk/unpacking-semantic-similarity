"""
  The code is modified from https://github.com/karpathy/nanoGPT
"""
import json
import inspect
from typing import Self

import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from scipy.stats import pearsonr, spearmanr

class ModelConfig:
    def __init__(self, in_dim=1024, out_dim=512, dropout=0.2, bias=False, **kwargs):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.bias = bias
        for key, value in kwargs.items():
            setattr(self, key, value)

class SimilarityLearner(nn.Module):
    def __init__(self, config:ModelConfig|None = None):
        super().__init__()

        if config:
            self.config = config
            
            modules = dict(
                drop = nn.Dropout(config.dropout),
                linear = nn.Linear(config.in_dim, config.out_dim, bias=config.bias),
            )

            self.module_dict = nn.ModuleDict(modules)
            self.apply(self._init_weights)
        # otherwise assume the model will be loaded from a file


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, a:torch.Tensor, b:torch.Tensor, targets:torch.Tensor=None):
        """
          a, b    - torch.Tensor of text embeddings
          targets - torch.Tensor of a similarity measure between a and b
        """
        # Dropout automatically respects self.training mode
        f_a = self.module_dict['drop'](a)
        f_b = self.module_dict['drop'](b)
        proj_a = self.module_dict['linear'](f_a)
        proj_b = self.module_dict['linear'](f_b)
        ab_similarity = F.cosine_similarity(proj_a, proj_b, dim=1)

        loss = None
        if targets is not None:
            loss = F.mse_loss(ab_similarity, targets.reshape(-1))

        return ab_similarity, loss

    def predict(self, a, b, targets=None, include_baseline:bool=False) -> dict:
        """
        Predict similarity scores for pairs of embeddings.
        If targets are provided, also compute the loss.
        """
        data = self._numpy_to_tensor(a=a, b=b, targets=targets)
        self.eval()
        with torch.no_grad():
            projected_similarities, projected_loss = self.forward(data['a'], data['b'], data['targets'])

        result = {
            "projected": projected_similarities.cpu().numpy().tolist(),
            "projected_mse": projected_loss.cpu().numpy().tolist() if projected_loss is not None else None,
        }
        result["projected_mae"] = float(F.l1_loss(projected_similarities, data['targets'].reshape(-1)).cpu().numpy())
        pearson, _ = pearsonr(result["projected"], targets.reshape(-1))
        spearman, _ = spearmanr(result["projected"], targets.reshape(-1))
        result["projected_pearson"] = float(pearson)
        result["projected_spearman"] = float(spearman)

        if include_baseline:
            baseline_similarities = F.cosine_similarity(data['a'], data['b'], dim=1)
            baseline_loss = None
            if targets is not None:
                baseline_loss = F.mse_loss(baseline_similarities, data['targets'].reshape(-1))
            baseline_np = baseline_similarities.cpu().numpy()
            targets_np = targets.reshape(-1)
            result["baseline"] = baseline_np.tolist()
            result["baseline_mse"] = baseline_loss.cpu().numpy().tolist() if baseline_loss is not None else None
            result["baseline_mae"] = float(F.l1_loss(baseline_similarities, data['targets'].reshape(-1)).cpu().numpy())
            pearson, _ = pearsonr(baseline_np, targets_np)
            spearman, _ = spearmanr(baseline_np, targets_np)
            result["baseline_pearson"] = float(pearson)
            result["baseline_spearman"] = float(spearman)

            # improvement is increased correlation
            result['improvement_pearson'] = result['projected_pearson'] - result['baseline_pearson']
            result['improvement_spearman'] = result['projected_spearman'] - result['baseline_spearman']
            # improvement is reduced loss
            result['improvement_mse'] = result['baseline_mse'] - result['projected_mse']
            result['improvement_mae'] = result['baseline_mae'] - result['projected_mae']

        return result

    def _numpy_to_tensor(self, **kwargs) -> dict:
        device = next(self.parameters()).device

        data = dict(**kwargs)
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = torch.tensor(value, dtype=torch.float32, device=device)
            elif isinstance(value, torch.Tensor):
                data[key] = value.to(device)
        return data

    def evaluate_metrics(self, a, b, targets):
        """
        Compute evaluation metrics for model ranking.
        Returns dict with MSE, Pearson correlation, and Spearman correlation.
        """
        self.eval()  # Set the model to evaluation mode

        data = self._numpy_to_tensor(a=a, b=b, targets=targets)
        result = {}
        with torch.no_grad():
            similarities, loss = self.forward(data['a'], data['b'], data['targets'])
            
            # Convert to numpy for correlation calculations
            pred_np = similarities.cpu().numpy()
            target_np = data['targets'].cpu().numpy().reshape(-1)
            
            # Compute correlations
            pearson_corr, _ = pearsonr(pred_np, target_np)
            spearman_corr, _ = spearmanr(pred_np, target_np)
            
            result = {
                'pearson_correlation': float(pearson_corr),
                'spearman_correlation': float(spearman_corr),
                'mse_loss': float(loss.item()),
                # Mean Absolute Error (MAE)
                # MAE = (1/n) * Σ|predicted_i - actual_i|
                'mae': float(F.l1_loss(similarities, data['targets'].reshape(-1)).item())
            }
        
        # Calculate the cosine similarities between all unprojected a,b pairs
        # This computes pairwise cosine similarity for each row in the batch
        targets = data['targets'].reshape(-1)
        baseline_similarity = F.cosine_similarity(data['a'], data['b'], dim=1)
        baseline_mse = float(F.mse_loss(baseline_similarity, targets).item())
        baseline_mae = float(F.l1_loss(baseline_similarity, targets).item())

        # Correlate baseline similarities with target similarities
        baseline_similarity = baseline_similarity.cpu().numpy()
        baseline_pearson_corr, _ = pearsonr(baseline_similarity, target_np)
        baseline_spearman_corr, _ = spearmanr(baseline_similarity, target_np)

        result['baseline_pearson'] = float(baseline_pearson_corr)
        result['baseline_spearman'] = float(baseline_spearman_corr)
        result['baseline_mse'] = baseline_mse
        result['baseline_mae'] = baseline_mae
        # improvement is increased correlation
        result['improvement_pearson'] = result['pearson_correlation'] - result['baseline_pearson']
        result['improvement_spearman'] = result['spearman_correlation'] - result['baseline_spearman']
        # improvement is reduced loss
        result['improvement_mse'] = result['baseline_mse'] - result['mse_loss']
        result['improvement_mae'] = result['baseline_mae'] - result['mae']
        return result
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        # and filter out those that do not require grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # create optim groups. any parameters that is 2d will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # create adamw optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused adamw: {use_fused}")

        return optimizer

    def sanitize(self, x:dict) -> dict:
        """
        Remove any JSON-incompatible entries for JSON serialization.
        """
        sanitized = {}
        for key, value in x.items():
            if isinstance(value, (str, int, float, bool, list)) or value is None:
                sanitized[key] = value
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize(value)
        print(f"Sanitized config for saving: {sanitized}")
        return sanitized

    def save(self, filename:str, **kwargs):
        """
        Save the model configuration and state to a file.
        """
        # if no extension is given, use .json
        if not filename.endswith('.json'):
            filename += '.json'

        # save the model config and state
        torch.save(self.state_dict(), filename.replace('.json', '.pt'))
        
        # save the config and additional settings
        with open(filename, "w") as f:
            f.write(json.dumps(self.sanitize({"config": self.config.__dict__, **kwargs}), indent=4))

        print(f"Model config saved to '{filename}' and weights to '{filename.replace('.json', '.pt')}'")

    def load(self, filename:str) -> Self:
        """
        Load the model configuration and state from a file.
        """
        # load the config and additional settings first
        with open(filename, "r") as f:
            config_data = json.load(f)
            self.config = ModelConfig(**config_data.get("config", {}))
            
            # Initialize the model structure based on loaded config
            modules = dict(
                drop = nn.Dropout(self.config.dropout),
                linear = nn.Linear(self.config.in_dim, self.config.out_dim, bias=self.config.bias),
            )
            self.module_dict = nn.ModuleDict(modules)
            self.apply(self._init_weights)
            
            for key, value in config_data.items():
                if key != "config":
                    setattr(self, key, value)
        
        # load the model state
        state_dict = torch.load(filename.replace('.json', '.pt'))
        
        # Handle potential key mismatches in state_dict
        if any(key.startswith('module_dict.') for key in state_dict.keys()):
            # State dict has old format with 'module_dict.' prefix, load as is
            self.load_state_dict(state_dict)
        else:
            # Try to load with strict=False to handle key mismatches
            self.load_state_dict(state_dict, strict=False)
        
        print(f"Model loaded from '{filename}' and weights from '{filename.replace('.json', '.pt')}'")
        return self