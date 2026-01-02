# Save a BERTopic Model Bundle

Persist a trained BERTopic model to disk and store R-side extras in a
companion RDS file. This is the recommended way to reuse a model across
sessions when working through reticulate.

## Usage

``` r
save_bertopic_model(topic_model, path)
```

## Arguments

- topic_model:

  A list returned by
  [`train_bertopic_model()`](https://tpetric7.github.io/bertopicr/reference/train_bertopic_model.md).
  Must contain a Python BERTopic model at `topic_model$model`. Optional
  extras such as probabilities, reduced embeddings, topics over time, or
  topics per class are saved when present and set to `NULL` otherwise.

- path:

  Directory path to write the Python model to. The RDS companion file is
  saved as `paste0(path, "_extras.rds")`.

## Value

Invisibly returns `TRUE` after successful write.

## Examples

``` r
if (FALSE) { # \dontrun{
save_bertopic_model(topic_model, "topic_model")
} # }
```
