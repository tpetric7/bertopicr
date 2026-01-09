# Load and Reuse a BERTopic Model

This vignette shows how to load a previously saved BERTopic model in a
new session and reuse the extras stored alongside it. Set `eval = TRUE`
for the chunks you want to run.

## Load R packages

Python environment selection and checks are handled in the hidden setup
chunk at the top of the vignette.

``` r
library(reticulate)
library(bertopicr)
library(readr)
library(dplyr)
```

## GPU availability (optional)

``` r
reticulate::py_run_string(code = "import torch
print(torch.cuda.is_available())") # if GPU is available then TRUE else FALSE
```

## Load the model bundle

``` r
loaded <- load_bertopic_model("topic_model") # set the location of the model!
model <- loaded$model
extras <- loaded$extras
```

## Load data for inspection

``` r
sample_path <- system.file("extdata", "spiegel_sample.rds", package = "bertopicr")
df <- read_rds(sample_path)
docs <- df |> pull(text_clean)
```

## Create tables from the loaded model

``` r
doc_info <- get_document_info_df(model = model, texts = docs)
topic_info <- get_topic_info_df(model = model)
topics_df <- get_topics_df(model = model)
```

## Use extras and visualizations

``` r
visualize_barchart(model = model, filename = "barchart_demo")
visualize_distribution(
  model = model,
  text_id = 1,
  probabilities = extras$probabilities,
  filename = "vis_topic_dist_demo"
)
visualize_heatmap(model = model, filename = "vis_heat_demo")
visualize_topics(model = model, filename = "dist_map_demo")
```

``` r
visualize_documents(model = model, docs, reduced_embeddings = extras$reduced_embeddings_2d)
visualize_documents_2d(model = model, docs, reduced_embeddings = extras$reduced_embeddings_2d)
visualize_documents_3d(model = model, docs, reduced_embeddings = extras$reduced_embeddings_3d)
```

The following visualizations work only if *topics_over_time* and
*topics_per_class* were defined after model training or within the
[`train_bertopic_model()`](https://tpetric7.github.io/bertopicr/reference/train_bertopic_model.md)
function.

``` r
visualize_topics_over_time(model = model, topics_over_time_model = extras$topics_over_time)
visualize_topics_per_class(model, extras$topics_per_class, auto_open = FALSE)
```
