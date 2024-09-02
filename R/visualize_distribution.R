#' Visualize Topic Distribution for a Specific Document using BERTopic
#'
#' This function visualizes the topic distribution for a specific document from a BERTopic model using Python's Plotly library.
#' The visualization is saved as an interactive HTML file, which can be opened and viewed in a web browser.
#'
#' @param model A BERTopic model object. The model must have the method \code{visualize_distribution}.
#' @param text_id An integer specifying the index of the document for which the topic distribution is visualized. Default is 1.
#'                Must be a positive integer and a valid index within the \code{probabilities} matrix.
#' @param probabilities A matrix or data frame of topic probabilities, with rows corresponding to documents and columns to topics.
#'                      Each element represents the probability of a topic for a given document.
#' @return The function does not return a value but saves an HTML file containing the visualization
#'         and displays it in the current R environment.
#' @importFrom reticulate import
#' @importFrom readr read_file
#' @importFrom htmltools HTML browsable
#' @export
#' @examples
#' \dontrun{
#' # Assuming 'topic_model' is a BERTopic model object and 'probs' is a matrix of topic probabilities
#' visualize_distribution(model = topic_model, text_id = 1, probabilities = probs)
#' }
visualize_distribution <- function(model, text_id = 1, probabilities) {

  # Import Python modules using reticulate
  plotly <- tryCatch({
    reticulate::import("plotly")
  }, error = function(e) {
    stop("Failed to import plotly Python module. Ensure that plotly is installed in your Python environment.")
  })

  np <- tryCatch({
    reticulate::import("numpy")
  }, error = function(e) {
    stop("Failed to import numpy Python module. Ensure that numpy is installed in your Python environment.")
  })

  # Validate inputs
  if (missing(model)) {
    stop("The 'model' argument is missing. Please provide a BERTopic model object.")
  }

  if (missing(probabilities)) {
    stop("The 'probabilities' argument is missing. Please provide a matrix or data frame of topic probabilities.")
  }

  if (!is.numeric(text_id) || text_id <= 0 || text_id > nrow(probabilities)) {
    stop("'text_id' should be a positive integer and a valid index within the 'probabilities' matrix.")
  }

  # Validate the model object by checking for the required method
  if (!"visualize_distribution" %in% names(model)) {
    stop("The provided 'model' object does not contain the method 'visualize_distribution'. Please provide a valid BERTopic model object.")
  }

  # Convert topic distribution to a numpy array
  topic_distr <- tryCatch({
    as.numeric(probabilities[text_id, ])
  }, error = function(e) {
    stop("Error in extracting topic distribution: ", e$message)
  })

  topic_distr_np <- tryCatch({
    np$array(topic_distr)
  }, error = function(e) {
    stop("Failed to convert topic distribution to a numpy array: ", e$message)
  })

  # Visualize the topic-document distribution for the specified document
  fig <- tryCatch({
    model$visualize_distribution(topic_distr_np, min_probability = 0.0)
  }, error = function(e) {
    stop("Error in visualizing topic distribution: ", e$message)
  })

  # Save the figure as an HTML file
  tryCatch({
    plotly$offline$plot(fig, filename = "topic_dist_interactive.html", auto_open = FALSE)
  }, error = function(e) {
    stop("Failed to save the plot as an HTML file: ", e$message)
  })

  # Check for required R packages
  if (!requireNamespace("htmltools", quietly = TRUE)) {
    stop("The 'htmltools' package is not installed. Please install it using install.packages('htmltools').")
  }

  if (!requireNamespace("readr", quietly = TRUE)) {
    stop("The 'readr' package is not installed. Please install it using install.packages('readr').")
  }

  # Read the HTML file content as a single string
  html_content <- tryCatch({
    readr::read_file("topic_dist_interactive.html")
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

# # Example usage
# # Assuming 'topic_model' is a BERTopic model object and 'probs' is a matrix of topic probabilities
# visualize_distribution(model = topic_model, text_id = 1, probabilities = probs)
