# Declare global variables to avoid R CMD check notes
utils::globalVariables(c("topic_model"))

#' Visualize Topics per Class
#'
#' This function visualizes the distribution of topics per class using a pre-trained BERTopic model.
#' The visualization is generated using the Plotly Python package and displayed within an R environment.
#'
#' @param model A BERTopic model object. Default is 'topic_model'.
#' @param topics_per_class A data frame or list containing the topics per class data. Default is 'topics_per_class'.
#' @param start An integer specifying the starting index of the topics to visualize. Default is 0.
#' @param end An integer specifying the ending index of the topics to visualize. Default is 10.
#' @param filename A string specifying the name of the HTML file to save the visualization. Default is "topics_per_class".
#' @param auto_open A logical value indicating whether to automatically open the HTML file after saving. Default is TRUE.
#'
#' @return A Plotly visualization of the topics per class, displayed as an HTML file within the R environment.
#' @export
#'
#' @examples
#' \dontrun{
#' visualize_topics_per_class(model = topic_model,
#'                            topics_per_class = topics_per_class,
#'                            start = 0, end = 7,
#'                            filename = "plot",
#'                            auto_open = TRUE)
#' }
visualize_topics_per_class <- function(model = topic_model,
                                       topics_per_class = topics_per_class,
                                       start = 0, end = 10,
                                       filename = "topics_per_class",
                                       auto_open = TRUE) {

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

  # Import required Python modules using reticulate
  bertopic <- tryCatch({
    reticulate::import("bertopic")
  }, error = function(e) {
    stop("Failed to import the bertopic Python module. Ensure that bertopic is installed in your Python environment.")
  })

  plotly <- tryCatch({
    reticulate::import("plotly")
  }, error = function(e) {
    stop("Failed to import the plotly Python module. Ensure that plotly is installed in your Python environment.")
  })

  # Validate the BERTopic model object
  if (missing(model) || !"visualize_topics_per_class" %in% names(model)) {
    stop("The 'model' argument is missing or does not have the 'visualize_topics_per_class' method. Ensure the model is a valid BERTopic model object.")
  }

  # Validate the topics_per_class argument
  if (!is.list(topics_per_class) && !is.data.frame(topics_per_class)) {
    stop("The 'topics_per_class' argument must be a list or data frame containing topics per class information.")
  }

  # Validate start and end arguments
  if (!is.numeric(start) || !is.numeric(end) || start < 0 || end < 0) {
    stop("The 'start' and 'end' arguments must be non-negative numeric values.")
  }

  # Validate the filename argument
  if (!is.character(filename) || nchar(filename) == 0) {
    stop("The 'filename' argument must be a non-empty string.")
  }

  # Validate the auto_open argument
  if (!is.logical(auto_open)) {
    stop("The 'auto_open' argument must be a logical (TRUE or FALSE).")
  }

  # Generate the list of topics to visualize
  topics_list <- tryCatch({
    reticulate::py_eval(sprintf("list(range(%d, %d))", as.integer(start), as.integer(end)))
  }, error = function(e) {
    stop("Failed to generate the list of topics using Python: ", e$message)
  })

  # Generate the figure using the specified BERTopic model and topics range
  fig_classes <- tryCatch({
    model$visualize_topics_per_class(topics_per_class, topics = topics_list)
  }, error = function(e) {
    stop("Error generating the visualization: ", e$message)
  })

  # Save the figure as an HTML file using Plotly
  tryCatch({
    plotly$offline$plot(fig_classes, filename = paste0(filename, ".html"), auto_open = auto_open)
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
# visualize_topics_per_class(model = topic_model,
#                            topics_per_class = topics_per_class,
#                            start = 0, end = 7,
#                            filename = "plot",
#                            auto_open = TRUE)
