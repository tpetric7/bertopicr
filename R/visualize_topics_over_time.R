# Declare global variables to avoid R CMD check notes
utils::globalVariables(c("topic_model", "topics_over_time"))

#' Visualize Topics Over Time using BERTopic
#'
#' This function visualizes topics over time from a BERTopic model using Python's Plotly library.
#' The visualization is saved as an interactive HTML file, which can be opened and viewed in a web browser.
#'
#' @param model A BERTopic model object. The model must have the method \code{visualize_topics_over_time}.
#' @param topics_over_time_model A topics-over-time model object created using the BERTopic model.
#' @param top_n_topics An integer specifying the number of top topics to display in the visualization. Default is 20.
#'                     Must be a positive integer.
#' @param filename A character string specifying the name of the HTML file to save the visualization.
#'                 The default value is "topics_over_time". The filename should not contain illegal characters.
#' @return The function does not return a value but saves an HTML file containing the visualization
#'         and displays it in the current R environment.
#' @importFrom reticulate import
#' @importFrom readr read_file
#' @importFrom htmltools HTML browsable
#' @export
#' @examples
#' \dontrun{
#' # Assuming 'topics_over_time_model' is a BERTopic model object
#' visualize_topics_over_time(model = topic_model,
#'                            topics_over_time_model = topics_over_time,
#'                            top_n_topics = 5,
#'                            filename = "plot")
#' }
visualize_topics_over_time <- function(model,
                                       topics_over_time_model,
                                       top_n_topics = 20,
                                       filename = "topics_over_time") {

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

  # Import necessary Python modules using reticulate
  plotly <- tryCatch({
    reticulate::import("plotly")
  }, error = function(e) {
    stop("Failed to import plotly Python module. Ensure that plotly is installed in your Python environment.")
  })

  # Validate inputs
  if (missing(model)) {
    stop("The 'model' argument is missing. Please provide a BERTopic model object.")
  }

  # Check if the model has the required method
  if (!reticulate::py_has_attr(model, "visualize_topics_over_time")) {
    stop("The 'model' provided does not contain the method 'visualize_topics_over_time'. Please ensure it is a valid BERTopic model object.")
  }

  if (missing(topics_over_time_model)) {
    stop("The 'topics_over_time_model' argument is missing. Please provide a topics-over-time model object.")
  }

  if (!is.numeric(top_n_topics) || top_n_topics <= 0) {
    stop("'top_n_topics' should be a positive integer.")
  }

  # Ensure the filename has no illegal characters for file naming
  filename <- gsub("[^[:alnum:]_]", "_", filename)

  # Visualize topics over time using the BERTopic model
  fig <- tryCatch({
    model$visualize_topics_over_time(topics_over_time_model, top_n_topics = as.integer(top_n_topics))
  }, error = function(e) {
    stop("Error in visualizing topics over time: ", e$message)
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

  # Display the saved HTML file content
  tryCatch({
    htmltools::browsable(htmltools::HTML(html_content))
  }, error = function(e) {
    stop("Failed to display the HTML content: ", e$message)
  })
}

# # Example usage
# # Assuming topics_over_time_model is already created and valid
# visualize_topics_over_time(model = topic_model,
#                            topics_over_time_model = topics_over_time,
#                            top_n_topics = 5,
#                            filename = "plot")
