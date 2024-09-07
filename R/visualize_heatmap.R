#' Visualize Topic Similarity Heatmap using BERTopic
#'
#' This function visualizes the topic similarity heatmap of topics from a BERTopic model using Python's Plotly library.
#' The visualization is saved as an interactive HTML file, which can be opened and viewed in a web browser.
#'
#' @param model A BERTopic model object. The model must have the method \code{visualize_heatmap}.
#' @param filename A character string specifying the name of the HTML file to save the visualization.
#'                 The default value is "topics_similarity_heatmap". The filename should not contain illegal characters.
#'                 The `.html` extension is added automatically if not provided.
#' @param auto_open Logical. If TRUE, opens the HTML file after saving. Default is FALSE.
#' @return The function does not return a value but saves an HTML file containing the visualization
#'         and displays it in the current R environment.
#' @importFrom reticulate import
#' @importFrom readr read_file
#' @importFrom htmltools HTML browsable
#' @export
#' @examples
#' \dontrun{
#' # Assuming 'topic_model' is a BERTopic model object
#' visualize_heatmap(model = topic_model, filename = "topics_similarity_heatmap", auto_open = FALSE)
#' }
visualize_heatmap <- function(model, filename = "topics_similarity_heatmap", auto_open = FALSE) {

  # Error handling for required packages
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("The 'reticulate' package is required but not installed. Please install it using install.packages('reticulate').")
  }

  if (!requireNamespace("readr", quietly = TRUE)) {
    stop("The 'readr' package is required but not installed. Please install it using install.packages('readr').")
  }

  if (!requireNamespace("htmltools", quietly = TRUE)) {
    stop("The 'htmltools' package is required but not installed. Please install it using install.packages('htmltools').")
  }

  # Import Python modules using reticulate
  plotly <- tryCatch({
    reticulate::import("plotly")
  }, error = function(e) {
    stop("Failed to import the 'plotly' Python module. Ensure that plotly is installed in your Python environment.")
  })

  # Validate the model input
  if (missing(model)) {
    stop("The 'model' argument is missing. Please provide a BERTopic model object.")
  }

  # Check if the model has the required method
  if (!reticulate::py_has_attr(model, "visualize_heatmap")) {
    stop("The 'model' provided does not contain the method 'visualize_heatmap'. Please ensure it is a valid BERTopic model object.")
  }

  # Ensure the filename has no illegal characters for file naming
  filename <- gsub("[^[:alnum:]_]", "_", filename)

  # Ensure the filename has the .html extension
  if (!grepl("\\.html$", filename)) {
    filename <- paste0(filename, ".html")
  }

  # Visualize topics using the BERTopic model
  fig <- tryCatch({
    model$visualize_heatmap(custom_labels = FALSE)
  }, error = function(e) {
    stop("Error in visualizing the heatmap: ", e$message)
  })

  # Save the figure as an HTML file
  tryCatch({
    plotly$offline$plot(fig, filename = filename, auto_open = auto_open)
  }, error = function(e) {
    stop("Failed to save the plot as an HTML file: ", e$message)
  })

  # Read the HTML file content as a single string
  html_content <- tryCatch({
    readr::read_file(filename)
  }, error = function(e) {
    stop("Failed to read the saved HTML file: ", e$message)
  })

  # Display the saved HTML file content in the R environment
  tryCatch({
    htmltools::browsable(htmltools::HTML(html_content))
  }, error = function(e) {
    stop("Failed to display the HTML content: ", e$message)
  })
}

# Example usage
# visualize_heatmap(model = topic_model, filename = "plot", auto_open = TRUE)
