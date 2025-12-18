# Visualize Topics Over Time using BERTopic

This function visualizes topics over time from a BERTopic model using
Python's Plotly library. The visualization is saved as an interactive
HTML file, which can be opened and viewed in a web browser.

## Usage

``` r
visualize_topics_over_time(
  model,
  topics_over_time_model,
  top_n_topics = 20,
  filename = "topics_over_time"
)
```

## Arguments

- model:

  A BERTopic model object. The model must have the method
  `visualize_topics_over_time`.

- topics_over_time_model:

  A topics-over-time model object created using the BERTopic model.

- top_n_topics:

  An integer specifying the number of top topics to display in the
  visualization. Default is 20. Must be a positive integer.

- filename:

  A character string specifying the name of the HTML file to save the
  visualization. The default value is "topics_over_time". The filename
  should not contain illegal characters.

## Value

The function does not return a value but saves an HTML file containing
the visualization and displays it in the current R environment.

## Examples

``` r
if (FALSE) { # \dontrun{
# Assuming 'topics_over_time_model' is a BERTopic model object
visualize_topics_over_time(model = topic_model,
                           topics_over_time_model = topics_over_time,
                           top_n_topics = 5,
                           filename = "plot")
} # }
```
