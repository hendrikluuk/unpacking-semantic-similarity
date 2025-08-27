"""
 Use the sentence transformer to embed the input text using the gte-large-env1.5 model from Hugging Face.

 https://huggingface.co/sentence-transformers

 You can build an index of referents for each concept and save the index to a file using:

 from utils.embedder import Embedder
 embedder = Embedder()
 # for the first time, you can build the index for all concepts
 embedder.build_index()

You can search the index for the most similar referents to a query using:
embedder.search("expressed", n=3)
"""
import gc
import os
import time
import pickle
from copy import copy

import torch
import numpy as np

from transformers import pipeline
from sentence_transformers import SentenceTransformer

from .models import models
from .api_model import ApiModel

# smaller sequence length consumes less memory
# max length of sentences in out/sampled_propositions_1500.json 
# is 164 characers which is around 60 tokens
MAX_SEQ_LENGTH_TOKENS = 512

cache_template = {"data": [], "embeddings": None, "index": {}}

class Embedder:
    def __init__(self, model:str="gte-large-en-v1.5", cache_tag:str="embeddings", local_files_only=True):
        self.model_args = {'local_files_only': local_files_only}
        self.model_factory(model)

        self.cache_path = os.path.join("cache", model, f"{cache_tag}.pkl")
        self.cache = copy(cache_template)

        if self.cache_path:
            self.load()

    def __del__(self):
        del self.model
        del self.cache
        torch.cuda.empty_cache()
        torch.mps.empty_cache()
        gc.collect()

    def model_factory(self, model: str):
        """
        Factory method to create a model instance based on the model name.
        """
        self.model_specs = models.get(model, {})
        model_id = self.model_specs.get("id")
        assert self.model_specs and model_id, f"Model '{model}' not found in the models dictionary."

        if torch.cuda.is_available():
            self.model_args["device"] = "cuda"
        elif torch.backends.mps.is_available():
            self.model_args["device"] = "mps"
        else:
            self.model_args["device"] = "cpu"
        # sadly needed to run most models
        self.model_args['trust_remote_code'] = True

        if self.model_specs.get("supports_sentence_transformer"):
            # Note! the model will be downloaded to ~/.cache/huggingface/hub folder
            self.model = SentenceTransformer(model_id, **self.model_args)
            self.model.max_seq_length = MAX_SEQ_LENGTH_TOKENS
        elif self.model_specs.get("supports_api"):
            self.model = ApiModel(model_id)
        else:
            # in some cases the model might not support SentenceTransformer interface
            self.model = pipeline("text-generation", model=model_id, **self.model_args)

    def embed(self, text: str, query:bool=False, preprocess:bool=True) -> np.array:
        if preprocess:
            text = self.preprocess(text)
        cache_hit = self.cache["index"].get(text)

        if cache_hit is not None:
            # return the cached embedding
            return self.cache["embeddings"][cache_hit]
        print(f"Warning: cache missed for '{text}'")

        if type(self.model).__name__ == "SentenceTransformer":
            encode_args = {key: self.model_specs[key] for key in ['prompt_name', 'prompt', 'query_prompt'] if key in self.model_specs}
            if query and "query_prompt" in encode_args:
                encode_args['prompt'] = encode_args["query_prompt"]
                del encode_args['query_prompt']
            return self.model.encode(text, **encode_args)
        elif type(self.model).__name__ == "ApiModel":
            return self.model.embed([text])

        # embed using a Transformers pipeline
        messages = map(lambda t: {"role": "user", "content": t}, text)
        return self.model(messages, return_tensors=True, truncation=True, padding=True)

    def embed_batch(self, texts: list[str], query:bool=False, preprocess:bool=True) -> tuple[list[str], np.array]:
        """
        Embed a batch of texts and store the embeddings in the cache if {key} is provided.
        """
        if preprocess:
            texts = [self.preprocess(text) for text in texts]
        if type(self.model).__name__ == "SentenceTransformer":
            prompt = None
            if query:
                prompt = self.model_specs.get("query_prompt")
            embeddings = self.model.encode(texts, prompt=prompt)
        elif type(self.model).__name__ == "ApiModel":
            embeddings = self.model.embed(texts)
        else:
            messages = map(lambda t: {"role": "user", "content": t}, texts)
            embeddings = self.model(messages, return_tensors=True, truncation=True, padding=True)
        return texts, embeddings

    def build_index(self, data:list[dict], rebuild:bool=False, export:bool=True):
        """
        Index the referents of each concept with embeddings and save the index.
        """
        if os.path.isfile(self.cache_path):
            print(f"Cache file '{self.cache_path}' already exists. Loading the index from cache.")
            self.load()
            if not rebuild:
                return

        if not os.path.exists(os.path.dirname(self.cache_path)):
            os.makedirs(os.path.dirname(self.cache_path))

        self.cache = copy(cache_template)
        self.cache["data"] = data

        start = time.time()

        print(f"Indexing {len(data)} items by 'a' and 'b' attributes ...", end=" ")
        texts = [self.preprocess(pair[index]) for pair in data for index in ["a", "b"] if pair.get(index)]
        texts = list(set(texts))  # remove duplicates
        texts, embeddings = self.embed_batch(texts)
        self.cache["embeddings"] = embeddings
        self.cache["index"] = {text: i for i, text in enumerate(texts)}

        elapsed = time.time() - start
        print(f"done in {elapsed:.3f} seconds.")

        if export:
            self.export()

    def preprocess(self, text: str) -> str:
        """
        Remove extraneous content such as SOP annotation symbols.
        """
        return text.strip().replace("[", "").replace("]", "")

    def export(self):
        """
        Export the cache to a file.
        """
        if self.cache:
            # make cache_path folder if it does not exist
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.cache, f)

    def load(self):
        """
        Load the cache from a file.
        """
        if os.path.isfile(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.cache = pickle.load(f)

    def similarity(self, text1:str, text2:str) -> float:
        """
        Calculate the similarity between two texts using their embeddings.
        Returns a float value representing the cosine similarity.
        """
        embeddings = []
        for text in [text1, text2]:
            embeddings.append(self.embed(text, preprocess=True))
                
        #if type(self.model).__name__ == "SentenceTransformer":
        #    return self.model.similarity(*embeddings)

        # calculate cosine similarity
        return self.cosine_similarity(*embeddings)    

    def cosine_similarity(self, embedding1:np.array, embedding2:np.array) -> float:
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))    

    def search(self, query:str, n:int=3, return_index:bool=False) -> list[str|int]:
        """
        Search the cache for {query} and return {n} most similar texts
        based on increasing embedding distance from the {query}.
        """
        if not self.cache["texts"]:
            return []

        query = self.preprocess(query)
        query_embedding = self.embed(query)
        
        distances = []
        for batch_index in range(len(self.cache["embeddings"])):
            # use cosine distance to find the most similar embeddings
            distances.append(np.linalg.norm(self.cache["embeddings"][batch_index].reshape((-1, len(query_embedding))) - query_embedding, axis=1))

        distances = np.concatenate(distances)
        indices = np.argsort(distances)
        if return_index:
            # return the indices of the most similar texts
            return indices[:n]

        if len(self.cache["texts"]) == len(indices):
            # if the number of texts is equal to the number of indices, we can use the indices directly
            texts = self.cache["texts"]
        else:
            texts = [text for batch in self.cache["texts"] for text in batch]

        return [texts[i] for i in indices[:n]]
