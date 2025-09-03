## Unpacking the suitcase of semantic similarity

This code base supports the eponymous manuscript that establishes algorithms for semantic entailment similarity quantification on concept and proposition levels, constructs a dataset with well-defined semantic entailment relationships, benchmarks 15 SOTA embedding models for accuracy in semantic entailment detection and estimates the contribution of semantic entailment to cosine similarity in text embeddings.

In order to run the code, you must adapt it to ensure connectivity to LLM and embedding APIs. You will also need to download local copies of embedding models from the Hugging Face Hub. The instructions below should give a high-level view of how to recapitulate the process.

#### Preliminaries

As this code base was targeted against an internal LLM API, you might need to spend some effort to establish the connectivity to LLMs and embedding models. You have to provide credentials in the environment, baseurl and endpoints for the API and adjust request/response format.

While the Github copilot can help you in doing that, here are some bread crumbs towards solving the puzzle:

```python
"""
utils/call_llm.py
"""
base_url = f"TODO:REPLACE"
user_id = os.getenv('LLM_API_USER_ID')

"""
utils/api_model.py
"""
baseurl = "TODO:REPLACE WITH EMBEDDING API URL"

"""
download_pubmed_abstracts.py
"""
# PubMed wants to know who you are before letting you download thousands of abstracts
email_address = os.getenv('PUBMED_ACCOUNT_EMAIL')
```



#### Harvesting of synonymous and parent-child concept labels

1. Harvest synonymous labels of conceptual entities from the LLM responses in the "limited-list-referents" task from the "probing concepts" study (relevant files are provided in external_resources folder).

   ```bash
   ./harvest_equivalent_concepts.py
   ```

   

3. Validate synonyms as being reasonably common and, if possible, unambiguously associated with the same conceptual entity across various contexts.

   ```bash
   # uses "validate concept labels" prompt template
   ./validate_equivalent_concepts.py
   ```

   

4. Harvest concept labels from parent (category label) and child (category members) entities from concepts with nested sets of referents in the "probing concepts" study 

   ```bash
   ./harvest_parent_child_concepts.py
   ```

   

#### Construction of Relational Domains

1. Download 10k abstracts from the PubMed

   ```bash
   ./download_pubmed_abstracts.py
   ```

2. Extract predicates to reduce the complexity of input

   ```bash
   # uses "extract predicates from abstract" prompt template
   ./extract_predicates.py
   ```

3. Cluster predicates into semantically related sets

   ```bash
   # uses "identify relational domains" prompt template
   ./cluster_predicates.py
   ```

4. Merge clusters that contain related predicates using an LLM agent that reasons about predicates and can execute code to get a preliminary set of relational domains.

   ```bash
   # uses "semantic agent" prompt template
   ./run_agent.py
   ```

5. Do a round of manual curation to yield a human curated set

7. Use an LLM to check the human curated set for inconsistencies (using prompt template "relational-domain-consistency") which you can resolve manually depending on whether you agree with the LLM or not



#### Construction of Propositions

1. Sample concept pairs (as uniformly as possible) from the restricted vocabulary with the following semantic entailment relationships: equivalent conceptual entities, parent-child conceptual entities, unrelated conceptual entities, equivalent predicates, contradictory predicates, unrelated predicates.

   ```bash
   ./sample_concepts.py
   ```

2. Generate 2500 propositions from the sampled concept pairs.

   ```bash
   # using prompt template "generate propositions"
   ./generate_propositions.py
   ```

3. Annotate propositions for symbolic and semantic entailment similarity and checked for validity (deterministically).

   ```bash
   ./annotate_propositions.py
   ```



#### Training linear projections

1. Sample data for training

```bash
# will store the data in out/sampled_concepts_2500.json
./sample_concepts.py
```

```bash
# will store the data in out/sampled_propositions_1500.json
./sample_propositions.py
```

2. Embed samples

```bash
# embed out/sampled_concepts*.json and out/sampled_propositions*.json datasets
./embed_samples.py
# embed trial.json
./embed_samples.py --pattern annotations/trial.json
```

3. Train projections

```bash
# will store results in out/train_results<out_dim>.json
./train.sh
```



#### Summarization of results

1. Summarize results as an Excel table

```bash
# will store results in ./train_results_summary.xlsx
./summarize_train_results.py
```

2. Compare the deviation of predictions by domain between certain model pairs

```bash
# stores all comparisons in out/prediction_error_comparison.json
./compare_deltas.sh
```

3. Identify proposition construction schemas where the estimates of semantic entailment are most divergent between the compared models

```bash
# use 'out/prediction_error_comparison.json' as input to estimate differential error significance
# stores the output in out/entailment_categories_with_differential_error_significance.csv
./get_entailment_categories_with_high_error_differential.py
```

4. Perform similarity prediction on the validation dataset

```bash
# stores the predictions in 'out/cosine_similarity_trial.json'
./estimate_cosine_similarity.py
```

5. Decompose the validation predictions into components

```bash
# will store results in './trial_summary.xlsx'
./regression_analysis_of_cosine_similarity.py
```
