# Visualize Documents in Reduced Embedding Space

This function generates a visualization of documents using a pre-trained
BERTopic model. It uses UMAP to reduce the dimensionality of embeddings
and Plotly for interactive visualizations.

## Usage

``` r
visualize_documents(
  model = topic_model,
  texts = texts_cleaned,
  reduced_embeddings = reduced_embeddings,
  custom_labels = FALSE,
  hide_annotation = TRUE,
  filename = "visualize_documents",
  auto_open = FALSE
)
```

## Arguments

- model:

  A BERTopic model object. Default is 'topic_model'.

- texts:

  A list or vector of cleaned text documents to visualize. Default is
  'texts_cleaned'.

- reduced_embeddings:

  A matrix of reduced-dimensionality embeddings. Typically generated
  using UMAP. Default is 'reduced_embeddings'.

- custom_labels:

  A logical value indicating whether to use custom labels for topics.
  Default is FALSE.

- hide_annotation:

  A logical value indicating whether to hide annotations in the plot.
  Default is TRUE.

- filename:

  A string specifying the name of the HTML file to save the
  visualization. Default is "visualize_documents".

- auto_open:

  A logical value indicating whether to automatically open the HTML file
  after saving. Default is FALSE.

## Value

A Plotly visualization of the documents, displayed as an HTML file
within the R environment.

## Examples

``` r
if (FALSE) { # \dontrun{
visualize_documents(model = topic_model,
                    texts = texts_cleaned,
                    reduced_embeddings = reduced_embeddings,
                    custom_labels = FALSE,
                    hide_annotation = TRUE,
                    filename = "visualize_documents",
                    auto_open = FALSE)
} # }
```
