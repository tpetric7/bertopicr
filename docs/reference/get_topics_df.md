# Get Topics DataFrame Function

This function retrieves all topics from a BERTopic model and converts
them into a data frame or tibble format.

## Usage

``` r
get_topics_df(model, return_tibble = TRUE)
```

## Arguments

- model:

  A BERTopic model object. Must be passed from the calling environment.

- return_tibble:

  Logical. If TRUE, returns a tibble. If FALSE, returns a data.frame.
  Default is TRUE.

## Value

A data.frame or tibble with columns for the word, score, and topic
number across all topics.

## Examples

``` r
if (FALSE) { # \dontrun{
topics_df <- get_topics_df(model = topic_model)
print(topics_df)
} # }
```
