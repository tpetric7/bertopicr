# Declare global variables to avoid R CMD check notes
utils::globalVariables(c("topic_model"))

#' Get Topics DataFrame Function
#'
#' This function retrieves all topics from a BERTopic model and converts them into a data frame or tibble format.
#'
#' @param model A BERTopic model object. Must be passed from the calling environment.
#' @param return_tibble Logical. If TRUE, returns a tibble. If FALSE, returns a data.frame. Default is TRUE.
#' @return A data.frame or tibble with columns for the word, score, and topic number across all topics.
#' @importFrom tibble as_tibble
#' @importFrom rlang .data
#' @importFrom purrr map_chr map_dbl
#' @export
#' @examples
#' \dontrun{
#' topics_df <- get_topics_df(model = topic_model)
#' print(topics_df)
#' }
get_topics_df <- function(model, return_tibble = TRUE) {

  # Check if the model is provided
  if (missing(model)) {
    stop("A BERTopic model must be provided.")
  }

  # Initialize an empty list to store DataFrames for each topic
  topics_list <- list()

  # Get the topics dictionary from the BERTopic model
  topics_dict <- tryCatch({
    model$get_topics()
  }, error = function(e) {
    stop("Failed to retrieve topics from the model: ", e$message)
  })

  # Check if topics_dict is valid and has content
  if (is.null(topics_dict) || length(topics_dict) == 0) {
    stop("No topics found in the provided model.")
  }

  # Loop through the topics dictionary and convert each topic to a DataFrame
  for (topic_number in names(topics_dict)) {
    topic_list <- topics_dict[[topic_number]]
    if (length(topic_list) > 0) {
      # Inline function to convert topic list to DataFrame
      df <- tryCatch({
        words <- purrr::map_chr(topic_list, ~ as.character(.[[1]]))
        scores <- purrr::map_dbl(topic_list, ~ .[[2]])
        data.frame(Word = words, Score = scores)
      }, error = function(e) {
        warning(paste("Failed to convert topic list to data frame for topic", topic_number, ":", e$message))
        NULL  # Return NULL in case of error
      })

      if (!is.null(df)) {
        df$Topic <- as.numeric(topic_number)
        topics_list[[length(topics_list) + 1]] <- df
      }
    }
  }

  # Combine all DataFrames into one
  topics_df <- do.call(rbind, topics_list)

  # Return as tibble if specified
  if (return_tibble) {
    topics_df <- tibble::as_tibble(topics_df)
  }

  return(topics_df)
}

# # Example usage
# topics_df <- get_topics_df(model = topic_model)
# print(topics_df)
