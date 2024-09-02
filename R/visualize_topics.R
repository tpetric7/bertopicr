#' Visualize Topics using BERTopic
#'
#' This function visualizes the intertopic distance map of topics from a BERTopic model using Python's Plotly library.
#' The visualization is saved as an interactive HTML file, which can be opened and viewed in a web browser.
#'
#' @param model A BERTopic model object. The model must have the method \code{visualize_topics}.
#' @param filename A character string specifying the name of the HTML file to save the visualization.
#'                 The default value is "intertopic_distance_map". The filename should not contain illegal characters.
#' @return The function does not return a value but saves an HTML file containing the visualization
#'         and displays it in the current R environment.
#' @export
#' @examples
#' \dontrun{
#' # Assuming 'topic_model' is a BERTopic model object
#' visualize_topics(model = topic_model, filename = "plot")
#' }
visualize_topics <- function(model, filename = "intertopic_distance_map") {

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
  bertopic <- tryCatch({
    reticulate::import("bertopic")
  }, error = function(e) {
    stop("Failed to import bertopic Python module. Ensure that bertopic is installed in your Python environment.")
  })

  plotly <- tryCatch({
    reticulate::import("plotly")
  }, error = function(e) {
    stop("Failed to import plotly Python module. Ensure that plotly is installed in your Python environment.")
  })

  # Validate inputs
  if (missing(model)) {
    stop("The 'model' argument is missing. Please provide a BERTopic model object.")
  }

  # Ensure the filename has no illegal characters for file naming
  filename <- gsub("[^[:alnum:]_]", "_", filename)

  # Validate the model object by checking for the required method
  if (!"visualize_topics" %in% names(model)) {
    stop("The provided 'model' object does not contain the method 'visualize_topics'. Please provide a valid BERTopic model object.")
  }

  # Visualize topics using the BERTopic model
  fig <- tryCatch({
    model$visualize_topics(custom_labels = FALSE)
  }, error = function(e) {
    stop("Error in visualizing topics: ", e$message)
  })

  # Save the figure as an HTML file
  tryCatch({
    plotly$offline$plot(fig, filename = paste0(filename, ".html"), auto_open = FALSE)
  }, error = function(e) {
    stop("Failed to save the plot as an HTML file: ", e$message)
  })

  # Read the HTML file content as a single string
  html_content <- tryCatch({
    readr::read_file(paste0(filename, ".html"))
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
# Assuming 'topic_model' is a BERTopic model object
# visualize_topics(model = topic_model, filename = "plot")
