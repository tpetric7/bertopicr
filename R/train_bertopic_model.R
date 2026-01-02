#' Train a BERTopic Model
#'
#' This function creates embeddings with sentence-transformers, configures UMAP,
#' HDBSCAN, and CountVectorizer, optionally wires a representation model, and fits
#' a BERTopic model from R. The returned model can be used with bertopicr helpers.
#'
#' @param docs Character vector of documents to model.
#' @param embedding_model Sentence-transformers model name or local path.
#' @param embeddings Optional precomputed embeddings (matrix or array).
#' @param embedding_batch_size Batch size for embedding encoding.
#' @param embedding_show_progress Logical. Show embedding progress bar.
#' @param umap_model Optional pre-built UMAP Python object. If NULL, one is created.
#' @param umap_n_neighbors Number of neighbors for UMAP.
#' @param umap_n_components Number of UMAP components.
#' @param umap_min_dist UMAP min_dist parameter.
#' @param umap_metric UMAP metric.
#' @param umap_random_state Random state for UMAP.
#' @param hdbscan_model Optional pre-built HDBSCAN Python object. If NULL, one is created.
#' @param hdbscan_min_cluster_size HDBSCAN min_cluster_size.
#' @param hdbscan_min_samples HDBSCAN min_samples.
#' @param hdbscan_metric HDBSCAN metric.
#' @param hdbscan_cluster_selection_method HDBSCAN cluster selection method.
#' @param hdbscan_gen_min_span_tree HDBSCAN gen_min_span_tree.
#' @param hdbscan_prediction_data Logical. Whether to generate prediction data.
#' @param hdbscan_core_dist_n_jobs HDBSCAN core_dist_n_jobs.
#' @param vectorizer_model Optional pre-built CountVectorizer Python object.
#' @param stop_words Stop words for CountVectorizer. Use "all_stopwords" to load the
#'   bundled multilingual list, "english", or a character vector.
#' @param ngram_range Length-2 integer vector for n-gram range.
#' @param min_df Minimum document frequency for CountVectorizer.
#' @param max_df Maximum document frequency for CountVectorizer.
#' @param max_features Maximum features for CountVectorizer.
#' @param strip_accents Passed to CountVectorizer. Use NULL to preserve umlauts.
#' @param decode_error Passed to CountVectorizer when decoding input bytes.
#' @param encoding Text encoding for CountVectorizer (defaults to "utf-8").
#' @param representation_model Representation model to use: "none", "keybert", "mmr", or "ollama".
#' @param representation_params Named list of parameters passed to the representation model.
#' @param ollama_model Ollama model name when representation_model = "ollama".
#' @param ollama_base_url Base URL for the Ollama OpenAI-compatible endpoint.
#' @param ollama_api_key API key placeholder for the Ollama OpenAI-compatible endpoint.
#' @param ollama_client_params Named list of extra parameters passed to openai$OpenAI().
#' @param ollama_prompt Optional prompt template for the Ollama OpenAI representation.
#' @param top_n_words Number of top words per topic to keep in the model.
#' @param calculate_probabilities Logical. Whether to calculate topic probabilities.
#' @param verbose Logical. Verbosity for BERTopic.
#' @param seed Optional random seed.
#' @param timestamps Optional vector of timestamps (Date/POSIXt/ISO strings or integer) for topics over time.
#'   Defaults to NULL (topics over time disabled).
#' @param topics_over_time_nr_bins Number of bins for topics_over_time.
#' @param topics_over_time_global_tuning Logical. Whether to enable global tuning for topics_over_time.
#' @param topics_over_time_evolution_tuning Logical. Whether to enable evolution tuning for topics_over_time.
#' @param classes Optional vector of class labels (character or factor) for topics per class.
#'   Defaults to NULL (topics per class disabled).
#' @param compute_reduced_embeddings Logical. If TRUE, computes 2D and 3D UMAP reductions.
#' @param reduced_embedding_n_neighbors Number of neighbors for reduced embeddings.
#' @param reduced_embedding_min_dist UMAP min_dist for reduced embeddings.
#' @param reduced_embedding_metric UMAP metric for reduced embeddings.
#' @param compute_hierarchical_topics Logical. If TRUE, computes hierarchical topics.
#' @param bertopic_args Named list of extra arguments passed to BERTopic().
#' @return A list with elements model, topics, probabilities, embeddings,
#'   reduced_embeddings_2d, reduced_embeddings_3d, hierarchical_topics,
#'   topics_over_time, and topics_per_class.
#' @export
#' @examples
#' \dontrun{
#' setup_python_environment()
#' texts <- c("Cats are great pets", "Dogs are loyal companions", "Markets fluctuate")
#' fit <- train_bertopic_model(texts, embedding_model = "sentence-transformers/all-MiniLM-L6-v2")
#' visualize_topics(fit$model, filename = "intertopic_distance_map", auto_open = FALSE)
#' }
train_bertopic_model <- function(
    docs,
    embedding_model = "Qwen/Qwen3-Embedding-0.6B",
    embeddings = NULL,
    embedding_batch_size = 32,
    embedding_show_progress = TRUE,
    umap_model = NULL,
    umap_n_neighbors = 15,
    umap_n_components = 5,
    umap_min_dist = 0.0,
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
    reduced_embedding_min_dist = 0.0,
    reduced_embedding_metric = "cosine",
    compute_hierarchical_topics = TRUE,
    bertopic_args = list()
) {

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("The 'reticulate' package is required but is not installed.")
  }

  if (missing(docs)) {
    stop("A character vector of documents must be provided.")
  }

  docs <- enc2utf8(as.character(docs))

  # --- SPREMEMBA: Odstranjeno "popravljanje" encodinga ---
  # Ker read_lines() že vrne pravilne UTF-8 znake, jih tukaj ne smemo
  # spreminjati z iconv(). Zaupamo vhodnim podatkom.

  docs_py <- reticulate::r_to_py(as.list(docs))

  representation_model <- match.arg(representation_model)
  representation_object <- NULL
  embedding_backend <- NULL
  if (!is.null(embedding_model) && !is.character(embedding_model)) {
    embedding_backend <- embedding_model
  }

  if (!is.null(seed)) {
    np <- reticulate::import("numpy")
    np$random$seed(as.integer(seed))
    umap_random_state <- as.integer(seed)
  }

  bertopic <- tryCatch(
    reticulate::import("bertopic"),
    error = function(e) {
      stop("Failed to import 'bertopic'. Ensure it is installed in your Python environment.")
    }
  )

  if (is.null(embeddings)) {
    if (is.null(embedding_model)) {
      stop("Provide either embedding_model or embeddings.")
    }
    sentence_transformers <- tryCatch(
      reticulate::import("sentence_transformers"),
      error = function(e) {
        stop("Failed to import 'sentence_transformers'.")
      }
    )
    if (is.null(embedding_backend)) {
      embedding_backend <- sentence_transformers$SentenceTransformer(embedding_model)
    }
    embeddings <- embedding_backend$encode(
      docs_py,
      batch_size = as.integer(embedding_batch_size),
      show_progress_bar = embedding_show_progress
    )
  } else if (representation_model == "keybert" && is.null(embedding_backend)) {
    if (is.null(embedding_model)) {
      stop("representation_model = 'keybert' requires embedding_model when embeddings are precomputed.")
    }
    if (is.character(embedding_model)) {
      sentence_transformers <- tryCatch(
        reticulate::import("sentence_transformers"),
        error = function(e) {
          stop("Failed to import 'sentence_transformers'.")
        }
      )
      embedding_backend <- sentence_transformers$SentenceTransformer(embedding_model)
    } else {
      embedding_backend <- embedding_model
    }
  }

  if (is.null(umap_model)) {
    umap <- tryCatch(
      reticulate::import("umap"),
      error = function(e) stop("Failed to import 'umap'.")
    )
    umap_model <- umap$UMAP(
      n_neighbors = as.integer(umap_n_neighbors),
      n_components = as.integer(umap_n_components),
      min_dist = umap_min_dist,
      metric = umap_metric,
      random_state = as.integer(umap_random_state)
    )
  }

  if (is.null(hdbscan_model)) {
    hdbscan <- tryCatch(
      reticulate::import("hdbscan"),
      error = function(e) stop("Failed to import 'hdbscan'.")
    )
    hdbscan_model <- hdbscan$HDBSCAN(
      min_cluster_size = as.integer(hdbscan_min_cluster_size),
      min_samples = as.integer(hdbscan_min_samples),
      metric = hdbscan_metric,
      cluster_selection_method = hdbscan_cluster_selection_method,
      gen_min_span_tree = hdbscan_gen_min_span_tree,
      prediction_data = hdbscan_prediction_data,
      core_dist_n_jobs = as.integer(hdbscan_core_dist_n_jobs)
    )
  }

  if (is.null(vectorizer_model)) {
    sklearn <- tryCatch(
      reticulate::import("sklearn"),
      error = function(e) stop("Failed to import scikit-learn.")
    )

    if (length(ngram_range) != 2) stop("ngram_range must be length-2 integer vector.")

    if (is.character(stop_words) && length(stop_words) == 1 && stop_words == "all_stopwords") {
      stopword_path <- system.file("extdata", "all_stopwords.txt", package = "bertopicr")
      if (file.exists(stopword_path)) {
        stop_words <- readr::read_lines(stopword_path)
      } else {
        warning("Stopword file not found. Proceeding without stop words.")
        stop_words <- NULL
      }
    }

    # --- SPREMEMBA: Tudi pri stop_words ne uporabljamo iconv ---
    # Če so prebrani z encoding="UTF-8", so že pravilni.

    if (!is.null(stop_words)) {
      stop_words <- enc2utf8(as.character(stop_words))
    }

    if (is.numeric(max_df) && length(max_df) == 1 && max_df > 1) {
      max_df <- as.integer(max_df)
    }

    vectorizer_model <- sklearn$feature_extraction$text$CountVectorizer(
      ngram_range = reticulate::tuple(as.integer(ngram_range[[1]]), as.integer(ngram_range[[2]])),
      stop_words = stop_words,
      min_df = as.integer(min_df),
      max_df = max_df,
      max_features = as.integer(max_features),
      strip_accents = strip_accents,
      decode_error = decode_error,
      encoding = encoding,
      # Ohranimo token_pattern za vsak slučaj, ker je varnejši za Unicode
      token_pattern = "(?u)\\b\\w+\\b"
    )
  }

  if (representation_model != "none") {
    representation <- tryCatch(
      reticulate::import("bertopic.representation"),
      error = function(e) stop("Failed to import 'bertopic.representation'.")
    )
    if (representation_model == "keybert") {
      representation_object <- do.call(representation$KeyBERTInspired, representation_params)
    }
    if (representation_model == "mmr") {
      if (length(representation_params) == 0) representation_params <- list(diversity = 0.3)
      representation_object <- do.call(representation$MaximalMarginalRelevance, representation_params)
    }
    if (representation_model == "ollama") {
      if (is.null(ollama_model)) stop("ollama_model required for 'ollama'.")
      if (is.null(ollama_prompt)) {
        ollama_prompt <- paste(
          "I have a topic that contains the following documents:",
          "[DOCUMENTS]",
          "The topic is described by the following keywords: [KEYWORDS]",
          "",
          "Based on the information above, extract a short but highly descriptive topic label of at most 5 words.",
          "Make sure it is in the following format:",
          "topic: <topic label>",
          sep = "\n"
        )
      }
      openai <- tryCatch(
        reticulate::import("openai"),
        error = function(e) stop("Failed to import 'openai'. Install the Python openai package.")
      )
      client_args <- c(
        list(base_url = ollama_base_url, api_key = ollama_api_key),
        ollama_client_params
      )
      client <- do.call(openai$OpenAI, client_args)
      if (is.null(representation_params$prompt)) representation_params$prompt <- ollama_prompt
      if (is.null(representation_params$chat)) representation_params$chat <- TRUE
      if (is.null(representation_params$exponential_backoff)) {
        representation_params$exponential_backoff <- TRUE
      }
      representation_params <- c(list(client), list(model = ollama_model), representation_params)
      representation_object <- do.call(representation$OpenAI, representation_params)
    }
  }

  bertopic_init <- list(
    umap_model = umap_model,
    hdbscan_model = hdbscan_model,
    vectorizer_model = vectorizer_model,
    representation_model = representation_object,
    calculate_probabilities = calculate_probabilities,
    top_n_words = as.integer(top_n_words),
    verbose = verbose
  )
  if (!is.null(embedding_backend)) {
    bertopic_init$embedding_model <- embedding_backend
  }
  bertopic_init <- utils::modifyList(bertopic_init, bertopic_args)

  model <- do.call(bertopic$BERTopic, bertopic_init)
  fit <- model$fit_transform(docs_py, embeddings)

  reduced_embeddings_2d <- NULL
  reduced_embeddings_3d <- NULL
  if (isTRUE(compute_reduced_embeddings)) {
    umap_reducer <- reticulate::import("umap")
    reduced_embeddings_2d <- umap_reducer$UMAP(
      n_neighbors = as.integer(reduced_embedding_n_neighbors),
      n_components = as.integer(2),
      min_dist = reduced_embedding_min_dist,
      metric = reduced_embedding_metric
    )$fit_transform(embeddings)

    reduced_embeddings_3d <- umap_reducer$UMAP(
      n_neighbors = as.integer(reduced_embedding_n_neighbors),
      n_components = as.integer(3),
      min_dist = reduced_embedding_min_dist,
      metric = reduced_embedding_metric
    )$fit_transform(embeddings)
  }

  hierarchical_topics <- NULL
  if (isTRUE(compute_hierarchical_topics)) {
    hierarchical_topics <- model$hierarchical_topics(docs_py)
  }

  topics_over_time <- NULL
  if (!is.null(timestamps)) {
    timestamps <- if (inherits(timestamps, "Date") || inherits(timestamps, "POSIXt")) {
      format(timestamps, "%Y-%m-%dT%H:%M:%S")
    } else if (is.numeric(timestamps)) {
      if (any(!is.finite(timestamps))) {
        stop("timestamps must not contain NA or infinite values.")
      }
      if (any(timestamps %% 1 != 0)) {
        stop("timestamps must be integer or Date/POSIXt/ISO 8601 strings.")
      }
      as.integer(timestamps)
    } else if (is.factor(timestamps)) {
      as.character(timestamps)
    } else if (is.character(timestamps)) {
      timestamps
    } else {
      stop("timestamps must be Date, POSIXt, character (ISO 8601), or integer.")
    }

    if (is.character(timestamps)) {
      iso_pattern <- "^\\d{4}-\\d{2}-\\d{2}(T\\d{2}:\\d{2}:\\d{2})?$"
      if (any(!grepl(iso_pattern, timestamps))) {
        stop("timestamps must be ISO 8601 strings (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS) or Date/POSIXt.")
      }
    }

    timestamps_py <- reticulate::r_to_py(as.list(timestamps))
    topics_over_time <- model$topics_over_time(
      docs_py,
      timestamps_py,
      nr_bins = as.integer(topics_over_time_nr_bins),
      global_tuning = topics_over_time_global_tuning,
      evolution_tuning = topics_over_time_evolution_tuning
    )
  }

  topics_per_class <- NULL
  if (!is.null(classes)) {
    if (is.factor(classes)) {
      classes <- as.character(classes)
    }
    if (!is.character(classes)) {
      stop("classes must be character or factor.")
    }
    if (length(classes) != length(docs)) {
      stop("classes must be the same length as docs.")
    }
    classes <- enc2utf8(classes)
    classes_py <- reticulate::r_to_py(as.list(classes))
    topics_per_class <- model$topics_per_class(docs_py, classes = classes_py)
  }

  list(
    model = model,
    topics = fit[[1]],
    probabilities = fit[[2]],
    embeddings = embeddings,
    reduced_embeddings_2d = reduced_embeddings_2d,
    reduced_embeddings_3d = reduced_embeddings_3d,
    hierarchical_topics = hierarchical_topics,
    topics_over_time = topics_over_time,
    topics_per_class = topics_per_class
  )
}
