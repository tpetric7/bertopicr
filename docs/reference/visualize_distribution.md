# Visualize Topic Distribution for a Specific Document using BERTopic

This function visualizes the topic distribution for a specific document
from a BERTopic model using Python's Plotly library. The visualization
is saved as an interactive HTML file, which can be opened and viewed in
a web browser.

## Usage

``` r
visualize_distribution(
  model,
  text_id = 1,
  probabilities,
  filename = "topic_dist_interactive",
  auto_open = FALSE
)
```

## Arguments

- model:

  A BERTopic model object. The model must have the method
  `visualize_distribution`.

- text_id:

  An integer specifying the index of the document for which the topic
  distribution is visualized. Default is 1. Must be a positive integer
  and a valid index within the `probabilities` matrix.

- probabilities:

  A matrix or data frame of topic probabilities, with rows corresponding
  to documents and columns to topics. Each element represents the
  probability of a topic for a given document.

- filename:

  A character string specifying the name of the HTML file to save the
  visualization. Default is "topic_dist_interactive". The .html
  extension will be added automatically.

- auto_open:

  Logical. If TRUE, the HTML file will automatically open in the
  browser. Default is FALSE.

## Value

The function does not return a value but saves an HTML file containing
the visualization and displays it in the current R environment.

## Examples

``` r
if (FALSE) { # \dontrun{
# Assuming 'topic_model' is a BERTopic model object and 'probs' is a matrix of topic probabilities
visualize_distribution(
  model = topic_model,
  text_id = 1,
  probabilities = probs,
  filename = "custom_filename",
  auto_open = TRUE)
} # }
```
