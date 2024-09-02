#' Find Topics DataFrame Function
#'
#' This function finds the most similar topics to given keywords using a BERTopic model
#' and returns the results in a data frame or tibble format.
#'
#' @param model A BERTopic model object. Must be passed from the calling environment.
#' @param queries A vector of keywords or phrases to query the topics for.
#' @param top_n Number of top similar topics to retrieve for each query. Default is 10.
#' @param return_tibble Logical. If TRUE, returns a tibble. If FALSE, returns a data.frame. Default is TRUE.
#' @return A data.frame or tibble with columns for the keyword, topics, and similarity scores for each query.
#' @export
#' @examples
#' # Example of finding similar topics using a BERTopic model
#' if (exists("topic_model")) {
#'   queries <- c("national minority", "minority issues", "nationality issues")
#'   find_topics_df(model = topic_model, queries = queries, top_n = 10)
#' } else {
#'   message("No topic_model found. Please load a BERTopic model and try again.")
#' }
find_topics_df <- function(model, queries, top_n = 10, return_tibble = TRUE) {
  # Import Python modules using reticulate
  bertopic <- tryCatch({
    reticulate::import("bertopic")
  }, error = function(e) {
    stop("Failed to import bertopic Python module. Ensure that bertopic is installed in your Python environment.")
  })

  # Check if the model and queries are provided
  if (missing(model)) {
    stop("A BERTopic model must be provided.")
  }
  if (missing(queries)) {
    stop("A vector of queries must be provided.")
  }

  # Ensure top_n is an integer
  top_n <- as.integer(top_n)

  # Initialize an empty list to store results
  results_list <- list()

  # Loop through each query and retrieve the top topics
  for (query in queries) {
    # Use reticulate to interact with the Python object
    similar_topics <- model$find_topics(query, top_n = top_n)

    # Create a temporary data frame for the current query
    similar_topics_df <- data.frame(
      keyword = query,
      topics = similar_topics[[1]],
      similarity = unlist(similar_topics[[2]])
    )

    # Append the result to the results list
    results_list[[query]] <- similar_topics_df
  }

  # Combine all results into a single data frame
  combined_results_df <- do.call(rbind, results_list)

  # Return as tibble if specified
  if (return_tibble) {
    combined_results_df <- tibble::as_tibble(combined_results_df)
  }

  return(combined_results_df)
}
