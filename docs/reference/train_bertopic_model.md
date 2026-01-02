# Train a BERTopic Model

This function creates embeddings with sentence-transformers, configures
UMAP, HDBSCAN, and CountVectorizer, optionally wires a representation
model, and fits a BERTopic model from R. The returned model can be used
with bertopicr helpers.

## Usage

``` r
train_bertopic_model(
  docs,
  embedding_model = "Qwen/Qwen3-Embedding-0.6B",
  embeddings = NULL,
  embedding_batch_size = 32,
  embedding_show_progress = TRUE,
  umap_model = NULL,
  umap_n_neighbors = 15,
  umap_n_components = 5,
  umap_min_dist = 0,
  umap_metric = "cosine",
  umap_random_state = 42,
  hdbscan_model = NULL,
  hdbscan_min_cluster_size = 50,
  hdbscan_min_samples = 20,
  hdbscan_metric = "euclidean",
  hdbscan_cluster_selection_method = "eom",
  hdbscan_gen_min_span_tree = TRUE,
  hdbscan_prediction_data = TRUE,
  hdbscan_core_dist_n_jobs = 1,
  vectorizer_model = NULL,
  stop_words = "all_stopwords",
  ngram_range = c(1, 3),
  min_df = 2L,
  max_df = 50L,
  max_features = 10000,
  strip_accents = NULL,
  decode_error = "strict",
  encoding = "UTF-8",
  representation_model = c("none", "keybert", "mmr", "ollama"),
  representation_params = list(),
  ollama_model = NULL,
  ollama_base_url = "http://localhost:11434/v1",
  ollama_api_key = "ollama",
  ollama_client_params = list(),
  ollama_prompt = NULL,
  top_n_words = 200L,
  calculate_probabilities = TRUE,
  verbose = TRUE,
  seed = NULL,
  timestamps = NULL,
  topics_over_time_nr_bins = 20L,
  topics_over_time_global_tuning = TRUE,
  topics_over_time_evolution_tuning = TRUE,
  classes = NULL,
  compute_reduced_embeddings = TRUE,
  reduced_embedding_n_neighbors = 10L,
  reduced_embedding_min_dist = 0,
  reduced_embedding_metric = "cosine",
  compute_hierarchical_topics = TRUE,
  bertopic_args = list()
)
```

## Arguments

- docs:

  Character vector of documents to model.

- embedding_model:

  Sentence-transformers model name or local path.

- embeddings:

  Optional precomputed embeddings (matrix or array).

- embedding_batch_size:

  Batch size for embedding encoding.

- embedding_show_progress:

  Logical. Show embedding progress bar.

- umap_model:

  Optional pre-built UMAP Python object. If NULL, one is created.

- umap_n_neighbors:

  Number of neighbors for UMAP.

- umap_n_components:

  Number of UMAP components.

- umap_min_dist:

  UMAP min_dist parameter.

- umap_metric:

  UMAP metric.

- umap_random_state:

  Random state for UMAP.

- hdbscan_model:

  Optional pre-built HDBSCAN Python object. If NULL, one is created.

- hdbscan_min_cluster_size:

  HDBSCAN min_cluster_size.

- hdbscan_min_samples:

  HDBSCAN min_samples.

- hdbscan_metric:

  HDBSCAN metric.

- hdbscan_cluster_selection_method:

  HDBSCAN cluster selection method.

- hdbscan_gen_min_span_tree:

  HDBSCAN gen_min_span_tree.

- hdbscan_prediction_data:

  Logical. Whether to generate prediction data.

- hdbscan_core_dist_n_jobs:

  HDBSCAN core_dist_n_jobs.

- vectorizer_model:

  Optional pre-built CountVectorizer Python object.

- stop_words:

  Stop words for CountVectorizer. Use "all_stopwords" to load the
  bundled multilingual list, "english", or a character vector.

- ngram_range:

  Length-2 integer vector for n-gram range.

- min_df:

  Minimum document frequency for CountVectorizer.

- max_df:

  Maximum document frequency for CountVectorizer.

- max_features:

  Maximum features for CountVectorizer.

- strip_accents:

  Passed to CountVectorizer. Use NULL to preserve umlauts.

- decode_error:

  Passed to CountVectorizer when decoding input bytes.

- encoding:

  Text encoding for CountVectorizer (defaults to "utf-8").

- representation_model:

  Representation model to use: "none", "keybert", "mmr", or "ollama".

- representation_params:

  Named list of parameters passed to the representation model.

- ollama_model:

  Ollama model name when representation_model = "ollama".

- ollama_base_url:

  Base URL for the Ollama OpenAI-compatible endpoint.

- ollama_api_key:

  API key placeholder for the Ollama OpenAI-compatible endpoint.

- ollama_client_params:

  Named list of extra parameters passed to openai\$OpenAI().

- ollama_prompt:

  Optional prompt template for the Ollama OpenAI representation.

- top_n_words:

  Number of top words per topic to keep in the model.

- calculate_probabilities:

  Logical. Whether to calculate topic probabilities.

- verbose:

  Logical. Verbosity for BERTopic.

- seed:

  Optional random seed.

- timestamps:

  Optional vector of timestamps (Date/POSIXt/ISO strings or integer) for
  topics over time. Defaults to NULL (topics over time disabled).

- topics_over_time_nr_bins:

  Number of bins for topics_over_time.

- topics_over_time_global_tuning:

  Logical. Whether to enable global tuning for topics_over_time.

- topics_over_time_evolution_tuning:

  Logical. Whether to enable evolution tuning for topics_over_time.

- classes:

  Optional vector of class labels (character or factor) for topics per
  class. Defaults to NULL (topics per class disabled).

- compute_reduced_embeddings:

  Logical. If TRUE, computes 2D and 3D UMAP reductions.

- reduced_embedding_n_neighbors:

  Number of neighbors for reduced embeddings.

- reduced_embedding_min_dist:

  UMAP min_dist for reduced embeddings.

- reduced_embedding_metric:

  UMAP metric for reduced embeddings.

- compute_hierarchical_topics:

  Logical. If TRUE, computes hierarchical topics.

- bertopic_args:

  Named list of extra arguments passed to BERTopic().

## Value

A list with elements model, topics, probabilities, embeddings,
reduced_embeddings_2d, reduced_embeddings_3d, hierarchical_topics,
topics_over_time, and topics_per_class.

## Examples

``` r
if (FALSE) { # \dontrun{
setup_python_environment()
texts <- c("Cats are great pets", "Dogs are loyal companions", "Markets fluctuate")
fit <- train_bertopic_model(texts, embedding_model = "sentence-transformers/all-MiniLM-L6-v2")
visualize_topics(fit$model, filename = "intertopic_distance_map", auto_open = FALSE)
} # }
```
