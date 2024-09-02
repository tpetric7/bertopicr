# Declare global variables to avoid R CMD check notes
utils::globalVariables(c("Representative_Docs", "RepresentativeDocs"))

#' Get Topic Information DataFrame
#'
#' This function retrieves topic information from a BERTopic model and processes it to unnest
#' list columns, replace NA values, and consolidate columns with the same prefix.
#'
#' @param model A BERTopic model object.
#' @param drop_expanded_columns Logical. If TRUE, drops the expanded columns after consolidation. Default is TRUE.
#' @return A data.frame or tibble with unnested and consolidated columns.
#' @importFrom dplyr rename rowwise mutate ungroup select all_of
#' @importFrom tidyr unnest_wider
#' @importFrom rlang sym .data :=
#' @export
#' @examples
#' \dontrun{
#' topic_info_df <- get_topic_info_df(model = topic_model,
#' drop_expanded_columns = TRUE)
#' print(topic_info_df)
#' }
get_topic_info_df <- function(model, drop_expanded_columns = TRUE) {

  # Check if the model is provided
  if (missing(model)) {
    stop("A BERTopic model must be provided.")
  }

  # Get topic info from the BERTopic model
  topic_info_df <- model$get_topic_info()

  # Temporarily rename conflicting columns if they exist
  if ("Representative_Docs" %in% names(topic_info_df)) {
    topic_info_df <- dplyr::rename(topic_info_df, RepresentativeDocs = .data$Representative_Docs)
  }

  # Identify list columns to unnest
  list_cols <- names(topic_info_df)[sapply(topic_info_df, is.list)]

  # Unnest the list columns using tidyr::unnest_wider
  for (col in list_cols) {
    topic_info_df <- tidyr::unnest_wider(topic_info_df, !!rlang::sym(col), names_sep = "_")
  }

  # Replace NA with an empty string across all columns
  topic_info_df[is.na(topic_info_df)] <- ""

  # Dynamically consolidate columns with the same prefix
  prefixes <- unique(gsub("_.*", "", list_cols))

  for (prefix in prefixes) {
    # General case for other prefixes
    prefix_cols <- grep(paste0("^", prefix, "_"), names(topic_info_df), value = TRUE)
    topic_info_df[prefix_cols] <- lapply(topic_info_df[prefix_cols], as.character)
    topic_info_df <- topic_info_df %>%
      dplyr::rowwise() %>%
      dplyr::mutate(
        !!rlang::sym(prefix) := paste(dplyr::c_across(dplyr::all_of(prefix_cols)), collapse = ", ")
      ) %>%
      dplyr::ungroup()
  }

  # Optionally drop the old expanded columns
  if (drop_expanded_columns) {
    for (prefix in prefixes) {
      prefix_cols <- grep(paste0("^", prefix, "_"), names(topic_info_df), value = TRUE)
      if (length(prefix_cols) > 0) {
        topic_info_df <- dplyr::select(topic_info_df, -dplyr::all_of(prefix_cols))
      }
    }
  }

  # Revert the temporary column name changes back to original names
  if ("RepresentativeDocs" %in% names(topic_info_df)) {
    topic_info_df <- dplyr::rename(topic_info_df, Representative_Docs = .data$RepresentativeDocs)
  }

  return(topic_info_df)
}

# # Example usage
# topic_info_df <- get_topic_info_df(model = topic_model,
#                                    drop_expanded_columns = TRUE)
# print(topic_info_df)
