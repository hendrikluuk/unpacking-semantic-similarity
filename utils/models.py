# https://huggingface.co/spaces/mteb/leaderboard

models = {
    "text-embedding-3-large": {
        "id": "text-embedding-3-large",
        "supports_api": True,
        "mteb_rank": 16,
    },

    "text-embedding-3-small": {
        "id": "text-embedding-3-small",
        "supports_api": True,
        "mteb_rank": 32
    },

    "cohere.embed-multilingual-v3": {
        "id": "cohere.embed-multilingual-v3",
        "model": "Cohere Embed Multilingual v3",
        "supports_api": True,
        "mteb_rank": 13,
    },

    "gte-large-en-v1.5": {
        "id": "Alibaba-NLP/gte-large-en-v1.5",
        "supports_sentence_transformer": True,
        "url": "https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5",
        "mteb_rank": None
    },

    "gte-base-en-v1.5": {
        "id": "Alibaba-NLP/gte-base-en-v1.5",
        "supports_sentence_transformer": True,
        "url": "https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5",
        "mteb_rank":199 
    },

    "Qwen3-Embedding-8B": {
        "id": "Qwen/Qwen3-Embedding-8B",
        "supports_sentence_transformer": True,
        "url": "https://huggingface.co/Qwen/Qwen3-Embedding-8B",
        "mteb_rank": 2,
        #"query_prompt": "query"
    },

    "Qwen3-Embedding-0.6B": {
        "id": "Qwen/Qwen3-Embedding-0.6B",
        "supports_sentence_transformer": True,
        "url": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
        "mteb_rank": 4,
        #"query_prompt": "query"
    },

    "gte-Qwen2-7B-instruct": {
        "id": "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "supports_sentence_transformer": True,
        "url": "https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct",
        "mteb_rank": 6,
        #"query_prompt": "query"
    },

    "gte-Qwen2-1.5B-instruct": {
        "id": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "supports_sentence_transformer": True,
        "url": "https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "mteb_rank": 14,
        #"query_prompt": "query"
    },

    "e5-mistral-7b-instruct": {
        "id": "intfloat/e5-mistral-7b-instruct",
        "supports_sentence_transformer": True,
        "url": "https://huggingface.co/intfloat/e5-mistral-7b-instruct",
        "mteb_rank": 12,
        "prompt_name": "sts_query",
    },

    "bilingual-embedding-large": {
        "id": "Lajavaness/bilingual-embedding-large",
        "supports_sentence_transformer": True,
        "url": "https://huggingface.co/Lajavaness/bilingual-embedding-large",
        "mteb_rank": 15,
    },

    "GIST-large-Embedding-v0": {
        "id": "avsolatorio/GIST-large-Embedding-v0",
        "supports_sentence_transformer": True,
        "url": "https://huggingface.co/avsolatorio/GIST-large-Embedding-v0",
        "mteb_rank": 55,
    },

    "stella_en_1.5B_v5": {
        "id": "NovaSearch/stella_en_1.5B_v5",
        "supports_sentence_transformer": True,
        "url": "https://huggingface.co/NovaSearch/stella_en_1.5B_v5",
        "query_prompt": "s2s_query",
        "mteb_rank": 18
    },

    "bge-m3": {
        "id": "BAAI/bge-m3",
        "supports_sentence_transformer": True,
        "url": "https://huggingface.co/BAAI/bge-m3",
        "mteb_rank": 22
    },

    "snowflake-arctic-embed-l-v2.0": {
        "id": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "supports_sentence_transformer": True,
        "url": "https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0",
        "mteb_rank": 34
    },

    "amazon.titan-embed-text-v1": {
        "id": "amazon.titan-embed-text-v1",
        "model": "Titan Text Embeddings v1",
        "supports_api": True,
        "disabled": True,
        "mteb_rank": None
    },

    "amazon.titan-embed-g1-text-02": {
        "id": "amazon.titan-embed-g1-text-02",
        "model": "Titan Text Embeddings v2",
        "supports_api": True,
        "disabled": True,
        "mteb_rank": 200
    },

}

models = {model: models[model] for model in models if not models[model].get("disabled", False)}
