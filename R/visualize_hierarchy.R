# Declare global variables to avoid R CMD check notes
utils::globalVariables(c("topic_model"))

#' Visualize Topic Hierarchy Nodes using BERTopic
#'
#' This function visualizes the hierarchical clustering of topics from a BERTopic model.
#' If a hierarchical topics DataFrame is provided, it uses this for visualization; otherwise, it visualizes directly from the model.
#' The visualization is saved as an interactive HTML file, which can be opened and viewed in a web browser.
#'
#' @param model A BERTopic model object. The model must have the method \code{visualize_hierarchy}.
#' @param hierarchical_topics Optional. A hierarchical topics DataFrame created using the BERTopic model's \code{hierarchical_topics} method.
#'                            If provided, this object is used to generate the hierarchy visualization.
#' @param filename A character string specifying the name of the HTML file to save the visualization.
#'                 The default value is "topic_hierarchy". The filename should not contain illegal characters.
#' @param auto_open Logical. If \code{TRUE}, the HTML file will be opened automatically after being saved. Default is \code{TRUE}.
#' @return The function does not return a value but saves an HTML file containing the visualization
#'         and displays it in the current R environment.
#' @importFrom reticulate import
#' @importFrom readr read_file
#' @importFrom htmltools HTML browsable
#' @export
#' @examples
#' \dontrun{
#' # Assuming 'topic_model' is a BERTopic model object
#' visualize_hierarchy(model = topic_model, filename = "topic_hierarchy",
#' auto_open = TRUE)
#'
#' # Alternatively, provide a pre-calculated hierarchical_topics object
#' visualize_hierarchy(model = topic_model,
#' hierarchical_topics = hierarchical_topics,
#' filename = "topic_hierarchy",
#' auto_open = TRUE)
#' }
visualize_hierarchy <- function(model, hierarchical_topics = NULL, filename = "topic_hierarchy", auto_open = TRUE) {

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

  # Check if the model has the required method
  if (!reticulate::py_has_attr(model, "visualize_hierarchy")) {
    stop("The provided 'model' object does not contain the method 'visualize_hierarchy'. Please provide a valid BERTopic model object.")
  }

  if (!is.logical(auto_open)) {
    stop("'auto_open' should be a logical value (TRUE or FALSE).")
  }

  # Ensure the filename has no illegal characters for file naming
  filename <- gsub("[^[:alnum:]_]", "_", filename)

  # Generate the hierarchical clustering visualization
  fig_hierarchy <- tryCatch({
    if (!is.null(hierarchical_topics)) {
      # Use provided hierarchical_topics for visualization
      model$visualize_hierarchy(hierarchical_topics = hierarchical_topics, custom_labels = FALSE)
    } else {
      # Use the model directly to generate hierarchy visualization
      model$visualize_hierarchy(custom_labels = FALSE)
    }
  }, error = function(e) {
    stop("Error in generating hierarchical clustering visualization: ", e$message)
  })

  if (is.null(fig_hierarchy)) {
    stop("Failed to generate a hierarchy visualization. Please check your hierarchical_topics input.")
  }

  # Save the figure as an HTML file
  tryCatch({
    plotly$offline$plot(fig_hierarchy, filename = paste0(filename, ".html"), auto_open = auto_open)
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

# # Example usage
# visualize_hierarchy(model = topic_model,
#                     filename = "topic_hierarchy",
#                     auto_open = TRUE)
# visualize_hierarchy(model = topic_model,
#                     hierarchical_topics = hierarchical_topics,
#                     filename = "topic_hierarchy",
#                     auto_open = TRUE)
