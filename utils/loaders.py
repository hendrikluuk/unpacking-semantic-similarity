import os
import json

def load_helper(file_path: str) -> list[dict]:
    """
    Load a JSON file and return its content as a list of dictionaries.
    If the file does not exist, return an empty list.
    """
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, "r") as f:
        data = json.load(f)
    
    return data

def load_clusters(file_path: str = "out/clustered_predicates.json") -> list[list[dict]]:
    """
    Load clustered predicates from a JSON file and return them as a list.
    """
    return load_helper(file_path)

def load_equivalent_concepts(file_path: str = "annotations/equivalent_concepts.json") -> list[dict]:
    """
    Load equivalent concepts from a JSON file and return them as a list.
    """
    return load_helper(file_path)

def load_relational_domains(file_path: str = "annotations/relational_domains.json") -> list[list[dict]]:
    """
    Load relational domains from a JSON file and return them as a list.
    """
    return load_helper(file_path)

def load_related_concepts(file_path: str = "annotations/parent_child_concepts.json") -> list[dict]:
    """
    Load related concepts from a JSON file and return them as a list.
    """
    return load_helper(file_path)

def load_proposition_constraints(file_path: str = "annotations/proposition_similarity_constraints.json", format:str="for llm") -> dict:
    """
    Load related concepts from a JSON file and return them as a list.
    """
    result = load_helper(file_path)
    assert format in result, f"Format '{format}' not found in proposition constraints."
    return result[format]

def load_propositions(file_path: str = "annotations/propositions.json") -> list[dict]:
    """
    Load propositions from a JSON file and return them as a list.
    If the file does not exist, return an empty list.
    """
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, "r") as f:
        data = json.load(f)
    
    return data