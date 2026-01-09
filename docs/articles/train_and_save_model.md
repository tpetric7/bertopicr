# Train and Save a BERTopic Model

This vignette shows how to train a BERTopic model from R and persist it
to disk along with the R-side extras (probabilities, reduced embeddings,
and dynamic topic outputs). Set `eval = TRUE` for the chunks you want to
run.

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

## Load sample data

Below, the German sample dataframe is used for topic analysis.

``` r
sample_path <- system.file("extdata", "spiegel_sample.rds", package = "bertopicr")
df <- read_rds(sample_path)
docs <- df |> pull(text_clean)
```

## Train the model

The
[`train_bertopic_model()`](https://tpetric7.github.io/bertopicr/reference/train_bertopic_model.md)
function is a convenience function. For more options / parameter
finetuning, see the other vignette (topics_spiegel.Rmd) or the Quarto
file (inst/extdata/topics_spiegel.qmd).

For more settings of the
[`train_bertopic_model()`](https://tpetric7.github.io/bertopicr/reference/train_bertopic_model.md)
function, check the help file.

``` r
topic_model <- train_bertopic_model(
  docs = docs,
  top_n_words = 50L, # set integer numbger of top words
  embedding_model = "Qwen/Qwen3-Embedding-0.6B", # choose your (multilingual) model from huggingface.co
  embedding_show_progress = TRUE,
  timestamps = df$date, # set this to NULL if not applicable with your data
  classes = df$genre, # set this to NULL if not applicable with your data
  representation_model = "keybert" # keyword generation for each topic
)
```

## Save the model and extras

> BERTopic - WARNING: When you use `pickle` to save/load a BERTopic
> model,please make sure that the environments in which you save and
> load the model are **exactly** the same. The version of BERTopic,its
> dependencies, and python need to remain the same.

``` r
save_bertopic_model(topic_model, "topic_model")
```
