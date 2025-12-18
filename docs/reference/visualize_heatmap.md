# Visualize Topic Similarity Heatmap using BERTopic

This function visualizes the topic similarity heatmap of topics from a
BERTopic model using Python's Plotly library. The visualization is saved
as an interactive HTML file, which can be opened and viewed in a web
browser.

## Usage

``` r
visualize_heatmap(
  model,
  filename = "topics_similarity_heatmap",
  auto_open = FALSE
)
```

## Arguments

- model:

  A BERTopic model object. The model must have the method
  `visualize_heatmap`.

- filename:

  A character string specifying the name of the HTML file to save the
  visualization. The default value is "topics_similarity_heatmap". The
  filename should not contain illegal characters. The `.html` extension
  is added automatically if not provided.

- auto_open:

  Logical. If TRUE, opens the HTML file after saving. Default is FALSE.

## Value

The function does not return a value but saves an HTML file containing
the visualization and displays it in the current R environment.

## Examples

``` r
if (FALSE) { # \dontrun{
# Assuming 'topic_model' is a BERTopic model object
visualize_heatmap(model = topic_model, filename = "topics_similarity_heatmap", auto_open = FALSE)
} # }
```
