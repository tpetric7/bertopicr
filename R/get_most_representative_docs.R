#' Get Most Representative Documents for a Specific Topic
#'
#' This function filters a given data frame to select the most representative documents
#' for a specified topic based on their probability scores. The documents are sorted by
#' relevance in descending order, and the top n documents are returned.
#'
#' @param df A data frame containing at least the columns 'Topic', 'Document', and 'probs'.
#' @param topic_nr An integer specifying the topic number to filter the documents.
#' @param n_docs An integer specifying the number of top representative documents to return. Defaults to 5.
#'
#' @return A vector of the most representative documents corresponding to the specified topic.
#' If the number of documents available is less than `n_docs`, all available documents are returned.
#'
#' @importFrom dplyr filter arrange select desc
#' @importFrom utils head
#' @importFrom rlang .data
#' @examples
#' \dontrun{
#' # Assuming `df_docs` is a data frame with columns `Topic`, `Document`, and `probs`
#' get_most_representative_docs(df_docs, topic_nr = 3, n_docs = 5)
#' }
#' @export
get_most_representative_docs <- function(df, topic_nr, n_docs = 5) {

  # Error handling: Check if df is a data frame
  if (!is.data.frame(df)) {
    stop("The input 'df' must be a data frame.")
  }

  # Error handling: Check if necessary columns are present in the data frame
  if (!all(c("Topic", "Document", "probs") %in% colnames(df))) {
    stop("The data frame must contain 'Topic', 'Document', and 'probs' columns.")
  }

  # Error handling: Check if topic_nr is a numeric value
  if (!is.numeric(topic_nr) || length(topic_nr) != 1) {
    stop("The 'topic_nr' must be a single numeric value.")
  }

  # Error handling: Check if n_docs is a numeric value and greater than 0
  if (!is.numeric(n_docs) || length(n_docs) != 1 || n_docs <= 0) {
    stop("The 'n_docs' must be a single numeric value greater than 0.")
  }

  # Filter the data frame to include only the specified topic
  df_filtered <- df %>%
    dplyr::filter(.data$Topic == topic_nr)

  # Error handling: Check if there are any documents available for the specified topic
  if (nrow(df_filtered) == 0) {
    warning("No documents found for the specified topic.")
    return(character(0)) # Return an empty character vector if no documents are found
  }

  # Select and order the documents by their relevance to the topic, descending
  top_docs <- df_filtered %>%
    dplyr::arrange(desc(.data$probs)) %>%
    dplyr::select(.data$Document) %>%
    head(n_docs)

  # Return the list of top representative documents
  return(top_docs$Document)
}

# # Example usage of the function
# # Create a data frame similar to df_docs from topic model results
# df_docs <- tibble::tibble(Topic = results$Topic,
#                           Document = results$text_clean,
#                           probs = results$Probability)
# sampled_representative_docs <- get_most_representative_docs(df_docs,
#                                                             topic_nr = 3,
#                                                             n_docs = 5)
# unique(sampled_representative_docs)
