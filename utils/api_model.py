"""
    Embedding model that can be access via API requests.
"""
import numpy as np
from requests import post

from utils.models import models

available_models = [key for key, model in models.items() if model.get("supports_api")]

class ApiModel:
    baseurl = "TODO:REPLACE WITH EMBEDDING API URL"

    def __init__(self, model_name: str):
        assert model_name in available_models, f"Model '{model_name}' is not available. Choose from {available_models}."
        self.model_name = model_name

    def fetch(self, texts: list[str], batch_size:int=50) -> list[list[float]]:
        """
        Fetch the most similar texts to the query using the specified model.
        """
        query = {
            "model": self.model_name,
            "docs": texts
        }

        result = []
        if len(texts) > batch_size:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                query['docs'] = batch
                response = post(url=self.baseurl, json=query)
                response = response.json()
                model = response.get('reason')
                result.extend(response.get('result', []))
        else:
            # single request for small number of texts
            query['docs'] = texts
            response = post(url=self.baseurl, json=query)
            response = response.json()
            result = response.get('result')
            model = response.get('reason')

        if not result or len(result) != len(texts) or model != self.model_name:
            raise ValueError(f"Failed to fetch {len(texts)} results for model '{model}': {response}")

        return result

    def embed(self, texts: list) -> np.array:
        """
        Embed a list of texts using the specified model.
        """
        result = self.fetch(texts)
        # convert result (a list of lists of floats) to numpy ndarray
        return np.array(result, dtype=np.float32)