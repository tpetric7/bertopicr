#' Visualize Documents in 3D Space using BERTopic
#'
#' This function generates a 3D visualization of documents using a pre-trained BERTopic model and UMAP dimensionality reduction.
#' It uses Plotly for interactive visualizations and saves the output as an HTML file.
#'
#' @param model A BERTopic model object. Default is 'topic_model'.
#' @param texts A character vector or list of cleaned text documents to visualize.
#' @param reduced_embeddings A matrix or data frame of reduced-dimensionality embeddings (3D). Typically generated using UMAP.
#' @param custom_labels Logical. If TRUE, custom topic labels are used. Default is FALSE.
#' @param hide_annotation Logical. If TRUE, hides annotations on the plot. Default is TRUE.
#' @param tooltips A character vector of tooltips for hover information. Default is c("Topic", "Name", "Probability", "Text").
#' @param filename A character string specifying the name of the HTML file to save the visualization. Default is "visualize_documents_3d".
#'                 The `.html` extension is automatically added if not provided.
#' @param auto_open Logical. If TRUE, opens the HTML file in the browser after saving. Default is FALSE.
#' @return The function does not return a value but saves an HTML file containing the visualization
#'         and displays it in the current R environment.
#' @importFrom reticulate import
#' @importFrom readr read_file
#' @importFrom htmltools HTML browsable
#' @export
#' @examples
#' \dontrun{
#' visualize_documents_3d(model = topic_model,
#'   texts = texts_cleaned,
#'   reduced_embeddings = embeddings,
#'   custom_labels = FALSE,
#'   hide_annotation = TRUE,
#'   filename = "plot",
#'   auto_open = TRUE)
#' }
visualize_documents_3d <- function(model, texts, reduced_embeddings, custom_labels = FALSE,
                                   hide_annotation = TRUE, tooltips = c("Topic", "Name", "Probability", "Text"),
                                   filename = "visualize_documents_3d", auto_open = FALSE) {

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

  # Import necessary Python modules using reticulate
  plotly <- tryCatch({
    reticulate::import("plotly")
  }, error = function(e) {
    stop("Failed to import the 'plotly' Python module. Ensure that plotly is installed in your Python environment.")
  })

  np <- tryCatch({
    reticulate::import("numpy")
  }, error = function(e) {
    stop("Failed to import the 'numpy' Python module. Ensure that numpy is installed in your Python environment.")
  })

  bertopic <- tryCatch({
    reticulate::import("bertopic")
  }, error = function(e) {
    stop("Failed to import the 'bertopic' Python module. Ensure that bertopic is installed in your Python environment.")
  })

  # Validate inputs
  if (missing(model)) {
    stop("The 'model' argument is missing. Please provide a BERTopic model object.")
  }

  if (!is.list(texts) && !is.vector(texts)) {
    stop("The 'texts' argument must be a list or vector of cleaned text documents.")
  }

  if (!is.matrix(reduced_embeddings) && !is.data.frame(reduced_embeddings)) {
    stop("The 'reduced_embeddings' argument must be a matrix or data frame of reduced-dimensionality embeddings.")
  }

  # Ensure reduced_embeddings has 3 columns for 3D plotting
  if (ncol(reduced_embeddings) != 3) {
    stop("The 'reduced_embeddings' argument must have 3 dimensions (3 columns) for 3D plotting.")
  }

  # Ensure the filename has the .html extension
  if (!grepl("\\.html$", filename)) {
    filename <- paste0(filename, ".html")
  }

  # Extract topic information from the model
  doc_info <- model$get_document_info(texts)
  topics <- doc_info$Topic

  # Prepare tooltips
  tooltips_final <- vector("character", length(topics))
  for (i in seq_along(topics)) {
    row_info <- doc_info[i, ]
    tooltips_final[i] <- paste(
      "Topic:", row_info$Topic, "<br>",
      "Name:", row_info$Name, "<br>",
      "Probability:", row_info$Probability, "<br>",
      "Text:", paste(substr(row_info$Document, 1, 60),
                     "<br>",
                     substr(row_info$Document, 61, 120),
                     "<br>",
                     substr(row_info$Document, 121, 180))
    )
  }

  # Create a 3D scatter plot using Plotly
  fig <- plotly$graph_objs$Figure()

  # Set hovermode to 'closest' to ensure tooltips appear relative to the hovered data point
  fig$update_layout(hovermode = "closest")

  # Sort unique topics and add traces for each
  unique_topics <- sort(unique(topics))
  for (topic_id in unique_topics) {
    topic_indices <- which(topics == topic_id)
    fig$add_trace(
      plotly$graph_objs$Scatter3d(
        x = reduced_embeddings[topic_indices, 1],
        y = reduced_embeddings[topic_indices, 2],
        z = reduced_embeddings[topic_indices, 3],
        mode = 'markers',
        marker = list(size = 3, opacity = 0.8),
        name = paste("Topic", topic_id),
        text = tooltips_final[topic_indices],
        hoverinfo = 'text',
        hoverlabel = list(
          # bgcolor = 'rgba(255, 255, 255, 0.01)',  # Semi-transparent background
          bordercolor = 'rgba(255, 255, 255, 0.0)',
          font = list(size = 10, color = 'black')
        )
      )
    )
  }

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
