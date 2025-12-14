# Visualize Topic Hierarchy Nodes using BERTopic

This function visualizes the hierarchical clustering of topics from a
BERTopic model. If a hierarchical topics DataFrame is provided, it uses
this for visualization; otherwise, it visualizes directly from the
model. The visualization is saved as an interactive HTML file, which can
be opened and viewed in a web browser.

## Usage

``` r
visualize_hierarchy(
  model,
  hierarchical_topics = NULL,
  filename = "topic_hierarchy",
  auto_open = TRUE
)
```

## Arguments

- model:

  A BERTopic model object. The model must have the method
  `visualize_hierarchy`.

- hierarchical_topics:

  Optional. A hierarchical topics DataFrame created using the BERTopic
  model's `hierarchical_topics` method. If provided, this object is used
  to generate the hierarchy visualization.

- filename:

  A character string specifying the name of the HTML file to save the
  visualization. The default value is "topic_hierarchy". The filename
  should not contain illegal characters.

- auto_open:

  Logical. If `TRUE`, the HTML file will be opened automatically after
  being saved. Default is `TRUE`.

## Value

The function does not return a value but saves an HTML file containing
the visualization and displays it in the current R environment.

## Examples

``` r
if (FALSE) { # \dontrun{
# Assuming 'topic_model' is a BERTopic model object
visualize_hierarchy(model = topic_model, filename = "topic_hierarchy",
auto_open = TRUE)

# Alternatively, provide a pre-calculated hierarchical_topics object
visualize_hierarchy(model = topic_model,
hierarchical_topics = hierarchical_topics,
filename = "topic_hierarchy",
auto_open = TRUE)
} # }
```
