# Find Topics DataFrame Function

This function finds the most similar topics to given keywords using a
BERTopic model and returns the results in a data frame or tibble format.

## Usage

``` r
find_topics_df(model, queries, top_n = 10, return_tibble = TRUE)
```

## Arguments

- model:

  A BERTopic model object. Must be passed from the calling environment.

- queries:

  A vector of keywords or phrases to query the topics for.

- top_n:

  Number of top similar topics to retrieve for each query. Default is
  10.

- return_tibble:

  Logical. If TRUE, returns a tibble. If FALSE, returns a data.frame.
  Default is TRUE.

## Value

A data.frame or tibble with columns for the keyword, topics, and
similarity scores for each query.

## Examples

``` r
# Example of finding similar topics using a BERTopic model
if (exists("topic_model")) {
  queries <- c("national minority", "minority issues", "nationality issues")
  find_topics_df(model = topic_model, queries = queries, top_n = 10)
} else {
  message("No topic_model found. Please load a BERTopic model and try again.")
}
#> No topic_model found. Please load a BERTopic model and try again.
```
