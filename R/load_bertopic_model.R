#' Load a BERTopic Model Bundle
#'
#' Load a BERTopic model saved with `save_bertopic_model()` along with its
#' companion RDS file containing R-side extras.
#'
#' @param path Directory path where the Python model was saved.
#' @param embedding_model Optional embedding model to pass through to
#'   `BERTopic$load()` when the embedding model is not serialized.
#' @return A list with two elements: `model` (the BERTopic model) and `extras`
#'   (the R-side data saved in the companion RDS file).
#' @export
#' @examples
#' \dontrun{
#' loaded <- load_bertopic_model("topic_model")
#' doc_info <- get_document_info_df(model = loaded$model, texts = docs)
#' }
load_bertopic_model <- function(path, embedding_model = NULL) {
  bertopic <- reticulate::import("bertopic")

  if (is.null(embedding_model)) {
    model <- bertopic$BERTopic$load(path)
  } else {
    model <- bertopic$BERTopic$load(path, embedding_model = embedding_model)
  }

  extras <- readRDS(paste0(path, "_extras.rds"))
  list(model = model, extras = extras)
}
