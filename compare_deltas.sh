#!/bin/bash

# compare pairs of sibling models (differing in size)
# model with better performance in semantic entailment is on the left
./compare_deltas.py text-embedding-3-large text-embedding-3-small --overwrite
./compare_deltas.py Qwen3-Embedding-8B Qwen3-Embedding-0.6B
./compare_deltas.py gte-Qwen2-1.5B-instruct gte-Qwen2-7B-instruct

# compare top performers in semantic vs symbolic similarity
# model with better performance in semantic entailment is on the left
./compare_deltas.py text-embedding-3-large snowflake-arctic-embed-l-v2.0
./compare_deltas.py e5-mistral-7b-instruct snowflake-arctic-embed-l-v2.0
./compare_deltas.py gte-Qwen2-1.5B-instruct snowflake-arctic-embed-l-v2.0
./compare_deltas.py text-embedding-3-large bge-m3
./compare_deltas.py e5-mistral-7b-instruct bge-m3
./compare_deltas.py gte-Qwen2-1.5B-instruct bge-m3

# results are stored as a list in out/model_comparison.json
