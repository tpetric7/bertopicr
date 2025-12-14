# Get Topic DataFrame Function

This function retrieves a specified number of words with high
probability for a given topic number from a BERTopic model and returns
the results in a data frame or tibble format.

## Usage

``` r
get_topic_df(model, topic_number = 0, top_n = 10, return_tibble = TRUE)
```

## Arguments

- model:

  A BERTopic model object. Must be passed from the calling environment.

- topic_number:

  The topic number for which words and scores are retrieved.

- top_n:

  Number of top words to retrieve for the specified topic. Default
  is 10. If greater than 10, it will be set to 10 as BERTopic returns a
  maximum of 10 words.

- return_tibble:

  Logical. If TRUE, returns a tibble. If FALSE, returns a data.frame.
  Default is TRUE.

## Value

A data.frame or tibble with columns for the word, score, and topic
number.

## Examples

``` r
if (FALSE) { # \dontrun{
# Example usage:
if (exists("topic_model")) {
  topic_df <- get_topic_df(model = topic_model, topic_number = 3, top_n = 5)
  print(topic_df)
} else {
  message("No topic_model found. Please load a BERTopic model and try again.")
}
} # }
```
