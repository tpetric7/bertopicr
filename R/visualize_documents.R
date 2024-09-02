# Declare global variables to avoid R CMD check notes
utils::globalVariables(c("texts_cleaned", "topic_model", "reduced_embeddings"))

#' Visualize Documents in Reduced Embedding Space
#'
#' This function generates a visualization of documents using a pre-trained BERTopic model.
#' It uses UMAP to reduce the dimensionality of embeddings and Plotly for interactive visualizations.
#'
#' @param model A BERTopic model object. Default is 'topic_model'.
#' @param texts A list or vector of cleaned text documents to visualize. Default is 'texts_cleaned'.
#' @param reduced_embeddings A matrix of reduced-dimensionality embeddings. Typically generated using UMAP. Default is 'reduced_embeddings'.
#' @param custom_labels A logical value indicating whether to use custom labels for topics. Default is FALSE.
#' @param hide_annotation A logical value indicating whether to hide annotations in the plot. Default is TRUE.
#' @param filename A string specifying the name of the HTML file to save the visualization. Default is "visualize_documents".
#' @param auto_open A logical value indicating whether to automatically open the HTML file after saving. Default is FALSE.
#'
#' @return A Plotly visualization of the documents, displayed as an HTML file within the R environment.
#' @importFrom reticulate import
#' @importFrom readr read_file
#' @importFrom htmltools HTML browsable
#' @export
#'
#' @examples
#' \dontrun{
#' visualize_documents(model = topic_model,
#'                     texts = texts_cleaned,
#'                     reduced_embeddings = reduced_embeddings,
#'                     custom_labels = FALSE,
#'                     hide_annotation = TRUE,
#'                     filename = "visualize_documents",
#'                     auto_open = FALSE)
#' }
visualize_documents <- function(model = topic_model,
                                texts = texts_cleaned,
                                reduced_embeddings = reduced_embeddings,
                                custom_labels = FALSE,
                                hide_annotation = TRUE,
                                filename = "visualize_documents",
                                auto_open = FALSE) {

  # Check for required R packages
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("The 'reticulate' package is required but not installed. Please install it using install.packages('reticulate').")
  }

  if (!requireNamespace("readr", quietly = TRUE)) {
    stop("The 'readr' package is required but not installed. Please install it using install.packages('readr').")
  }

  if (!requireNamespace("htmltools", quietly = TRUE)) {
    stop("The 'htmltools' package is required but not installed. Please install it using install.packages('htmltools').")
  }

  # Import required Python modules using reticulate
  plotly <- tryCatch({
    reticulate::import("plotly")
  }, error = function(e) {
    stop("Failed to import the plotly Python module. Ensure that plotly is installed in your Python environment.")
  })

  # Error handling for input parameters
  if (missing(model)) {
    stop("The 'model' argument is missing. Please provide a BERTopic model object.")
  }

  # Check if the model has the required method
  if (!reticulate::py_has_attr(model, "visualize_documents")) {
    stop("The 'model' provided does not contain the method 'visualize_documents'. Please ensure it is a valid BERTopic model object.")
  }

  if (!is.list(texts) && !is.vector(texts)) {
    stop("The 'texts' argument must be a list or vector containing cleaned text documents.")
  }

  if (!is.matrix(reduced_embeddings) && !is.data.frame(reduced_embeddings)) {
    stop("The 'reduced_embeddings' argument must be a matrix or data frame of reduced-dimensionality embeddings.")
  }

  if (!is.logical(custom_labels)) {
    stop("The 'custom_labels' argument must be a logical (TRUE or FALSE).")
  }

  if (!is.logical(hide_annotation)) {
    stop("The 'hide_annotation' argument must be a logical (TRUE or FALSE).")
  }

  if (!is.character(filename) || nchar(filename) == 0) {
    stop("The 'filename' argument must be a non-empty string.")
  }

  if (!is.logical(auto_open)) {
    stop("The 'auto_open' argument must be a logical (TRUE or FALSE).")
  }

  # Generate the figure using the specified BERTopic model
  fig_documents <- tryCatch({
    model$visualize_documents(texts,  # corrected this part to match original function usage
                              reduced_embeddings = reduced_embeddings,
                              custom_labels = custom_labels,
                              hide_annotations = hide_annotation)
  }, error = function(e) {
    stop("Error generating visualization: ", e$message)
  })

  # Save the figure as an HTML file using Plotly
  tryCatch({
    plotly$offline$plot(fig_documents, filename = paste0(filename, ".html"), auto_open = auto_open)
  }, error = function(e) {
    stop("Failed to save the plot as an HTML file: ", e$message)
  })

  # Read the HTML file content as a single string
  html_content <- tryCatch({
    readr::read_file(paste0(filename, ".html"))
  }, error = function(e) {
    stop("Failed to read the saved HTML file: ", e$message)
  })

  # Display the saved HTML file content using HTML and browsable functions
  tryCatch({
    htmltools::browsable(htmltools::HTML(html_content))
  }, error = function(e) {
    stop("Failed to display the HTML content: ", e$message)
  })
}

# # Example usage
# visualize_documents(model = topic_model,
#                     texts = texts_cleaned,
#                     reduced_embeddings = reduced_embeddings,
#                     custom_labels = FALSE,
#                     hide_annotation = TRUE,
#                     filename = "visualize_documents",
#                     auto_open = FALSE)
