# Visualize Topics using BERTopic

This function visualizes the intertopic distance map of topics from a
BERTopic model using Python's Plotly library. The visualization is saved
as an interactive HTML file, which can be opened and viewed in a web
browser.

## Usage

``` r
visualize_topics(
  model,
  filename = "intertopic_distance_map",
  auto_open = FALSE
)
```

## Arguments

- model:

  A BERTopic model object. The model must have the method
  `visualize_topics`.

- filename:

  A character string specifying the name of the HTML file to save the
  visualization. The default value is "intertopic_distance_map". The
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
visualize_topics(model = topic_model, filename = "plot", auto_open = TRUE)
} # }
```
