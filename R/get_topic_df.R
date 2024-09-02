#' Get Topic DataFrame Function
#'
#' This function retrieves a specified number of words with high probability for a given topic number
#' from a BERTopic model and returns the results in a data frame or tibble format.
#'
#' @param model A BERTopic model object. Must be passed from the calling environment.
#' @param topic_number The topic number for which words and scores are retrieved.
#' @param top_n Number of top words to retrieve for the specified topic. Default is 10.
#'              If greater than 10, it will be set to 10 as BERTopic returns a maximum of 10 words.
#' @param return_tibble Logical. If TRUE, returns a tibble. If FALSE, returns a data.frame. Default is TRUE.
#' @return A data.frame or tibble with columns for the word, score, and topic number.
#' @importFrom purrr map_chr map_dbl
#' @importFrom tibble as_tibble
#' @examples
#' \dontrun{
#' # Example usage:
#' if (exists("topic_model")) {
#'   topic_df <- get_topic_df(model = topic_model, topic_number = 3, top_n = 5)
#'   print(topic_df)
#' } else {
#'   message("No topic_model found. Please load a BERTopic model and try again.")
#' }
#' }
#' @export
get_topic_df <- function(model, topic_number = 0, top_n = 10, return_tibble = TRUE) {

  # Check if the model is provided
  if (missing(model)) {
    stop("A BERTopic model must be provided.")
  }

  # Check if the topic number is provided and is a non-negative integer
  if (!is.numeric(topic_number) || topic_number < 0) {
    stop("A valid topic number must be provided (non-negative integer).")
  }

  # Ensure top_n is an integer and not more than 10
  top_n <- as.integer(top_n)
  if (top_n > 10) {
    warning("top_n is greater than 10. BERTopic returns a maximum of 10 words; setting top_n to 10.")
    top_n <- 10
  }

  # Retrieve the topic word list from the model
  topic_wordlist <- model$get_topic(topic_number)

  # Check if the topic_wordlist is valid
  if (is.null(topic_wordlist) || length(topic_wordlist) == 0) {
    stop(paste("No words found for topic number", topic_number))
  }

  # Limit the results to top_n words
  words <- purrr::map_chr(topic_wordlist[1:top_n], ~ as.character(.[[1]]))
  scores <- purrr::map_dbl(topic_wordlist[1:top_n], ~ .[[2]])

  # Create the data frame
  topic_df <- data.frame(
    Word = words,
    Score = scores,
    Topic = as.integer(topic_number)
  )

  # Return as tibble if specified
  if (return_tibble) {
    topic_df <- tibble::as_tibble(topic_df)
  }

  return(topic_df)
}

# Example function to test get_topic_df only when desired
test_get_topic_df <- function() {
  if (exists("topic_model")) {
    topic_df <- get_topic_df(model = topic_model, topic_number = 3, top_n = 5)
    print(topic_df)
  } else {
    message("No topic_model found. Please load a BERTopic model and try again.")
  }
}
