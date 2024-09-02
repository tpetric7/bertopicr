#' Get Document Information DataFrame
#'
#' This function retrieves document information from a BERTopic model and processes it to unnest
#' list columns, replace NA values, and consolidate columns with the same prefix.
#'
#' @param model A BERTopic model object.
#' @param texts A character vector containing the preprocessed texts to be passed to the BERTopic model.
#' @param drop_expanded_columns Logical. If TRUE, drops the expanded columns after consolidation. Default is TRUE.
#' @return A data.frame or tibble with unnested and consolidated columns.
#' @export
#' @importFrom dplyr %>% rename rowwise mutate ungroup select all_of c_across
#' @importFrom tidyr unnest_wider
#' @importFrom rlang .data sym :=
#' @examples
#' \dontrun{
#' document_info_df <- get_document_info_df(model = topic_model,
#' texts = texts_cleaned, drop_expanded_columns = TRUE)
#' print(document_info_df)
#' }
get_document_info_df <- function(model, texts, drop_expanded_columns = TRUE) {
  # Check if the model and texts are provided
  if (missing(model) || missing(texts)) {
    stop("Both a BERTopic model and texts must be provided.")
  }

  # Get document info from the BERTopic model
  document_info_df <- model$get_document_info(texts)

  # Temporarily rename the "Representative_document" and "Representative_Docs" columns to avoid conflicts
  if ("Representative_document" %in% names(document_info_df)) {
    document_info_df <- dplyr::rename(document_info_df, Is_Representative_document = .data$Representative_document)
  }

  if ("Representative_Docs" %in% names(document_info_df)) {
    document_info_df <- dplyr::rename(document_info_df, RepresentativeDocs = .data$Representative_Docs)
  }

  # Identify list columns to unnest
  list_cols <- names(document_info_df)[sapply(document_info_df, is.list)]

  # Unnest the list columns using tidyr::unnest_wider
  for (col in list_cols) {
    document_info_df <- tidyr::unnest_wider(document_info_df, !!rlang::sym(col), names_sep = "_")
  }

  # Replace NA with an empty string across all columns
  document_info_df[is.na(document_info_df)] <- ""

  # Dynamically consolidate columns with the same prefix
  prefixes <- unique(gsub("_.*", "", list_cols))

  for (prefix in prefixes) {
    # General case for other prefixes
    prefix_cols <- grep(paste0("^", prefix, "_"), names(document_info_df), value = TRUE)
    document_info_df[prefix_cols] <- lapply(document_info_df[prefix_cols], as.character)
    document_info_df <- document_info_df %>%
      dplyr::rowwise() %>%
      dplyr::mutate(
        !!rlang::sym(prefix) := paste(dplyr::c_across(dplyr::all_of(prefix_cols)), collapse = ", ")
      ) %>%
      dplyr::ungroup()
  }

  # Optionally drop the old expanded columns
  if (drop_expanded_columns) {
    for (prefix in prefixes) {
      prefix_cols <- grep(paste0("^", prefix, "_"), names(document_info_df), value = TRUE)
      if (length(prefix_cols) > 0) {
        document_info_df <- dplyr::select(document_info_df, -dplyr::all_of(prefix_cols))
      }
    }
  }

  # Revert the temporary column name changes back to original names
  if ("Is_Representative_document" %in% names(document_info_df)) {
    document_info_df <- dplyr::rename(document_info_df, Representative_document = .data$Is_Representative_document)
  }

  if ("RepresentativeDocs" %in% names(document_info_df)) {
    document_info_df <- dplyr::rename(document_info_df, Representative_Docs = .data$RepresentativeDocs)
  }

  return(document_info_df)
}

# Example usage
# Assuming 'topic_model' is a BERTopic model object
# document_info_df <- get_document_info_df(model = topic_model, texts = texts_cleaned, drop_expanded_columns = TRUE)
# print(document_info_df)
