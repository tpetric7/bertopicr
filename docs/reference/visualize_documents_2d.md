# Visualize Documents in 2D Space using BERTopic

This function generates a 3D visualization of documents using a
pre-trained BERTopic model and UMAP dimensionality reduction. It uses
Plotly for interactive visualizations and saves the output as an HTML
file.

## Usage

``` r
visualize_documents_2d(
  model,
  texts,
  reduced_embeddings,
  custom_labels = FALSE,
  hide_annotation = TRUE,
  tooltips = c("Topic", "Name", "Probability", "Text"),
  filename = "visualize_documents_2d",
  auto_open = FALSE
)
```

## Arguments

- model:

  A BERTopic model object. Default is 'topic_model'.

- texts:

  A character vector or list of cleaned text documents to visualize.

- reduced_embeddings:

  A matrix or data frame of reduced-dimensionality embeddings (2D).
  Typically generated using UMAP.

- custom_labels:

  Logical. If TRUE, custom topic labels are used. Default is FALSE.

- hide_annotation:

  Logical. If TRUE, hides annotations on the plot. Default is TRUE.

- tooltips:

  A character vector of tooltips for hover information. Default is
  c("Topic", "Name", "Probability", "Text").

- filename:

  A character string specifying the name of the HTML file to save the
  visualization. Default is "visualize_documents_2d". The `.html`
  extension is automatically added if not provided.

- auto_open:

  Logical. If TRUE, opens the HTML file in the browser after saving.
  Default is FALSE.

## Value

The function does not return a value but saves an HTML file containing
the visualization and displays it in the current R environment.

## Examples

``` r
if (FALSE) { # \dontrun{
visualize_documents_2d(model = topic_model,
  texts = texts_cleaned,
  reduced_embeddings = embeddings,
  custom_labels = FALSE,
  hide_annotation = TRUE,
  filename = "plot",
  auto_open = TRUE)
} # }
```
