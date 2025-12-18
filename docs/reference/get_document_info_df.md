# Get Document Information DataFrame

This function retrieves document information from a BERTopic model and
processes it to unnest list columns, replace NA values, and consolidate
columns with the same prefix.

## Usage

``` r
get_document_info_df(model, texts, drop_expanded_columns = TRUE)
```

## Arguments

- model:

  A BERTopic model object.

- texts:

  A character vector containing the preprocessed texts to be passed to
  the BERTopic model.

- drop_expanded_columns:

  Logical. If TRUE, drops the expanded columns after consolidation.
  Default is TRUE.

## Value

A data.frame or tibble with unnested and consolidated columns.

## Examples

``` r
if (FALSE) { # \dontrun{
document_info_df <- get_document_info_df(model = topic_model,
texts = texts_cleaned, drop_expanded_columns = TRUE)
print(document_info_df)
} # }
```
