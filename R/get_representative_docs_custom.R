#' Get Representative Documents for a Specific Topic
#'
#' This function filters a given data frame to select a specified number of representative documents
#' from a particular topic. It uses random sampling to select the documents.
#'
#' @param df A data frame containing at least the columns 'Topic' and 'Document'.
#' @param topic_nr An integer specifying the topic number to filter the documents.
#' @param n_docs An integer specifying the number of documents to sample for the specified topic.
#'
#' @return A vector of sampled documents corresponding to the specified topic.
#'
#' @importFrom dplyr filter sample_n
#' @importFrom rlang .data
#' @examples
#' \dontrun{
#' # Assuming `df_docs` is a data frame with columns `Topic`, `Document`, and `probs`
#' get_representative_docs_custom(df_docs, topic_nr = 3, n_docs = 5)
#' }
#'
#' @export
get_representative_docs_custom <- function(df, topic_nr, n_docs) {

  # Error handling: Check if df is a data frame
  if (!is.data.frame(df)) {
    stop("The input 'df' must be a data frame.")
  }

  # Error handling: Check if necessary columns are present in the data frame
  if (!all(c("Topic", "Document") %in% colnames(df))) {
    stop("The data frame must contain 'Topic' and 'Document' columns.")
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

  # Randomly sample n_docs from the filtered data frame
  df_sampled <- df_filtered %>%
    dplyr::sample_n(min(n_docs, nrow(df_filtered))) # Ensure not to sample more than available

  # Return the list of sampled documents
  return(df_sampled$Document)
}

# # Example usage of the function
# # Create a data frame from the topic model results
# df_docs <- tibble::tibble(Topic = results$Topic,
#                           Document = results$text_clean,
#                           probs = results$Probability)
# # Use the function
# sampled_docs <- get_representative_docs_custom(df_docs,
#                                                topic_nr = 3,
#                                                n_docs = 5)
# unique(sampled_docs)
