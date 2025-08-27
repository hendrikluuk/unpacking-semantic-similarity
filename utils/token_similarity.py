import tiktoken
from difflib import SequenceMatcher
from typing import Tuple

def token_similarity(text1: str, text2: str, overlap_weight:float=0.5) -> Tuple[float, float]:
    """
    Calculate similarity between two strings based on their tokens.
    Returns tuple of (overlap_score, sequence_score).
    
    overlap_score: Measures token overlap regardless of position (0-1)
    sequence_score: Measures similarity considering token order (0-1)
    
    Example:
        "Hello world" vs "world Hello" would have:
        - High overlap_score (same tokens)
        - Lower sequence_score (different order)
    """
    # Initialize tokenizer
    enc = tiktoken.get_encoding("cl100k_base")  # GPT-4's encoding
    
    # Tokenize both strings
    tokens1 = enc.encode(text1)
    tokens2 = enc.encode(text2)
    
    # Calculate token overlap score using set intersection
    unique_tokens1 = set(tokens1)
    unique_tokens2 = set(tokens2)
    overlap = len(unique_tokens1.intersection(unique_tokens2))
    # ranges from 0 to 1
    overlap_score = overlap / len(unique_tokens1.union(unique_tokens2))
    
    # Calculate sequence similarity using SequenceMatcher
    # ranges from 0 to 1
    sequence_score = SequenceMatcher(None, tokens1, tokens2).ratio()
    
    return (overlap_weight * overlap_score + 
            (1 - overlap_weight) * sequence_score)
