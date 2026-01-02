library(readr)
library(dplyr)
library(wordcloud2)
library(reticulate)
library(bertopicr)

reticulate::use_virtualenv("c:/path/to/your/venv", required = TRUE)
reticulate::py_config()
reticulate::py_available()
reticulate::py_run_string(code = "import torch
print(torch.cuda.is_available())") # GPU available (TRUE or FALSE)

sample_path <- system.file("extdata", "spiegel_sample.rds", package = "bertopicr")
df <- read_rds(sample_path)
docs <- df |> pull(text_clean)
stopword_path <- system.file("extdata", "all_stopwords.txt", package = "bertopicr")
all_stopwords <- read_lines(stopword_path)

tictoc::tic()
topic_model <- train_bertopic_model(
  docs = docs,
  top_n_words = 50L,
  embedding_model = "Qwen/Qwen3-Embedding-0.6B",
  embedding_show_progress = TRUE,
  timestamps = df$date, # set this to NULL if not applicable with your data
  classes = df$genre, # set this to NULL if not applicable with your data
  representation_model = "keybert"
  )
tictoc::toc()

(doc_info <- get_document_info_df(model = topic_model$model, texts = docs))
(topic_info <- get_topic_info_df(topic_model$model))
get_most_representative_docs(doc_info |> rename(probs = Probability), topic_nr = 1, n_docs = 2)
get_topic_df(topic_model$model, topic_number = 1)

# doc_info |> write_csv("doc_info.csv")

visualize_barchart(model = topic_model$model, filename = "barchart_demo")
visualize_distribution(topic_model$model, text_id = 1, probabilities = topic_model$probabilities, filename = "vis_topic_dist_demo")
visualize_heatmap(topic_model$model, filename = "vis_heat_demo")
visualize_topics(model = topic_model$model, filename = "dist_map_demo")
visualize_hierarchy(model = topic_model$model, hierarchical_topics = NULL, filename = "vis_hiclust_basic_demo", auto_open = FALSE)

# reduced_embeddings_2d and reduced_embeddings_3d in topic_model already computed
visualize_documents(model = topic_model$model, texts = docs, reduced_embeddings = topic_model$reduced_embeddings_2d, filename = "vis_docs_demo")
visualize_documents_2d(model = topic_model$model, texts = docs, reduced_embeddings = topic_model$reduced_embeddings_2d, filename = "vis_docs_2d_demo")
visualize_documents_3d(model = topic_model$model, texts = docs, reduced_embeddings = topic_model$reduced_embeddings_3d, filename = "vis_docs_3d_demo")

# hierarchical_topics = topic_model$model$hierarchical_topics(docs)
visualize_hierarchy(model = topic_model$model, hierarchical_topics = topic_model$hierarchical_topics, filename = "vis_hiclust_demo", auto_open = FALSE)

# To be properly implemented in the train_bertopc_model function
# timestamps <- as.list(df$date)
# timestamps <- lapply(timestamps, function(x) {
#   format(x, "%Y-%m-%dT%H:%M:%S")  # ISO 8601 format
# })
# topics_over_time  <- topic_model$model$topics_over_time(docs, timestamps, nr_bins=20L, global_tuning=TRUE, evolution_tuning=TRUE)
visualize_topics_over_time(model = topic_model$model, topics_over_time_model = topic_model$topics_over_time, filename = "topics_time_demo")

# # To be properly implemented in the train_bertopc_model function
# classes = as.list(df$genre) # text types
# topics_per_class = topic_model$model$topics_per_class(docs, classes=classes)
visualize_topics_per_class(topic_model$model, topics_per_class = topic_model$topics_per_class, filename = "topic_class_demo")

source("inst/extdata/wordcloud2a.R")
df_wc <- get_topic_df(topic_model$model, topic_number = 0, top_n = 10)
wordcloud2a(df_wc, size = 0.3)

dfs_wc <- get_topics_df(model = topic_model$model) |> filter(Topic >= 0)
colorVector = rep(sample(rainbow(34), 34), each = 10, length.out=nrow(dfs_wc))
wordcloud2a(dfs_wc, size = 0.15, color = colorVector)

################
# after training
################

# Save model after training
topic_model$model$save("topic_model")  # writes a folder
saveRDS(list(
  probabilities = topic_model$probabilities,
  topics_over_time = topic_model$topics_over_time,
  topics_per_class = topic_model$topics_per_class,
  timestamps = df$date, # set this to NULL if not applicable with your data
  classes = df$genre, # set this to NULL if not applicable with your data
  reduced_embeddings_2d = topic_model$reduced_embeddings_2d,
  reduced_embeddings_3d = topic_model$reduced_embeddings_3d
), "topic_model_extras.rds")


# Load model in new session
bertopic <- reticulate::import("bertopic")
model <- bertopic$BERTopic$load("topic_model")
extras <- readRDS("topic_model_extras.rds")

# Create tables and visualizations
(doc_info <- get_document_info_df(model = model, texts = docs))
(topic_info <- get_topic_info_df(model))
get_most_representative_docs(doc_info |> rename(probs = Probability), topic_nr = 1, n_docs = 2)
get_topic_df(model, topic_number = 1)

visualize_barchart(model = model)
visualize_distribution(model, text_id = 1, probabilities = extras$probabilities)
visualize_heatmap(model)
visualize_topics(model)
visualize_hierarchy(model, extras$hierarchical_topics)
visualize_documents_2d(model = model, docs, reduced_embeddings = extras$reduced_embeddings_2d)
visualize_documents_3d(model = model, docs, reduced_embeddings = extras$reduced_embeddings_3d)
visualize_topics_over_time(model = model, topics_over_time_model = extras$topics_over_time)
visualize_topics_per_class(model, extras$topics_per_class)
