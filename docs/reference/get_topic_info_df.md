# Get Topic Information DataFrame

This function retrieves topic information from a BERTopic model and
processes it to unnest list columns, replace NA values, and consolidate
columns with the same prefix.

## Usage

``` r
get_topic_info_df(model, drop_expanded_columns = TRUE)
```

## Arguments

- model:

  A BERTopic model object.

- drop_expanded_columns:

  Logical. If TRUE, drops the expanded columns after consolidation.
  Default is TRUE.

## Value

A data.frame or tibble with unnested and consolidated columns.

## Examples

``` r
if (FALSE) { # \dontrun{
topic_info_df <- get_topic_info_df(model = topic_model,
drop_expanded_columns = TRUE)
print(topic_info_df)
} # }
```
