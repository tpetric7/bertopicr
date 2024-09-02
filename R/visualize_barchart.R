#' Visualize BERTopic Bar Chart
#'
#' This function visualizes the topics of a BERTopic model using Plotly and saves the output
#' as an interactive HTML file. It checks for required Python modules and allows for custom file naming.
#'
#' @param model A BERTopic model object. Must be passed from the calling environment.
#' @param filename A character string specifying the name of the HTML file to save the bar chart.
#'                 Default is "topics_topwords_interactive_barchart.html".
#' @param open_file Logical. If TRUE, opens the HTML file after saving. Default is TRUE.
#' @return Displays the interactive bar chart within the R environment and saves it as an HTML file.
#' @importFrom reticulate import
#' @importFrom readr read_file
#' @importFrom htmltools HTML browsable
#' @examples
#' \dontrun{
#' visualize_barchart(model = topic_model, filename = "custom_barchart.html",
#' open_file = TRUE)
#' }
#' @export
visualize_barchart <- function(model, filename = "topics_topwords_interactive_barchart.html", open_file = FALSE) {

  # Error handling for model input
  if (missing(model)) {
    stop("A BERTopic model must be provided.")
  }

  # Import necessary Python modules using reticulate
  plotly <- tryCatch({
    reticulate::import("plotly")
  }, error = function(e) {
    stop("Python module 'plotly' is not installed. Please install it using pip.")
  })

  # Visualization using BERTopic's visualize_barchart method
  fig <- tryCatch({
    model$visualize_barchart(custom_labels = FALSE)
  }, error = function(e) {
    stop("Failed to generate the barchart visualization: ", e$message)
  })

  # Save the figure as an HTML file
  tryCatch({
    plotly$offline$plot(fig, filename = filename, auto_open = open_file)
  }, error = function(e) {
    stop("Failed to save the plot as an HTML file: ", e$message)
  })

  # Error handling for required R packages
  if (!requireNamespace("htmltools", quietly = TRUE)) {
    stop("The 'htmltools' package is not installed. Please install it using install.packages('htmltools').")
  }

  if (!requireNamespace("readr", quietly = TRUE)) {
    stop("The 'readr' package is not installed. Please install it using install.packages('readr').")
  }

  # Read the HTML file content as a single string
  html_content <- tryCatch({
    readr::read_file(filename)
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
# viz <- visualize_barchart(model = topic_model,
#                           filename = "custom_barchart.html",
#                           open_file = TRUE)
