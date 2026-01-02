# Load a BERTopic Model Bundle

Load a BERTopic model saved with
[`save_bertopic_model()`](https://tpetric7.github.io/bertopicr/reference/save_bertopic_model.md)
along with its companion RDS file containing R-side extras.

## Usage

``` r
load_bertopic_model(path, embedding_model = NULL)
```

## Arguments

- path:

  Directory path where the Python model was saved.

- embedding_model:

  Optional embedding model to pass through to `BERTopic$load()` when the
  embedding model is not serialized.

## Value

A list with two elements: `model` (the BERTopic model) and `extras` (the
R-side data saved in the companion RDS file).

## Examples

``` r
if (FALSE) { # \dontrun{
loaded <- load_bertopic_model("topic_model")
doc_info <- get_document_info_df(model = loaded$model, texts = docs)
} # }
```
