#' Save a BERTopic Model Bundle
#'
#' Persist a trained BERTopic model to disk and store R-side extras in a
#' companion RDS file. This is the recommended way to reuse a model across
#' sessions when working through reticulate.
#'
#' @param topic_model A list returned by `train_bertopic_model()`. Must contain
#'   a Python BERTopic model at `topic_model$model`. Optional extras such as
#'   probabilities, reduced embeddings, topics over time, or topics per class
#'   are saved when present and set to `NULL` otherwise.
#' @param path Directory path to write the Python model to. The RDS companion
#'   file is saved as `paste0(path, "_extras.rds")`.
#' @return Invisibly returns `TRUE` after successful write.
#' @export
#' @examples
#' \dontrun{
#' save_bertopic_model(topic_model, "topic_model")
#' }
save_bertopic_model <- function(topic_model, path) {
  topic_model$model$save(path)

  get_optional_extra <- function(obj, name) {
    tryCatch(obj[[name]], error = function(e) NULL)
  }

  extras <- list(
    probabilities = get_optional_extra(topic_model, "probabilities"),
    topics_over_time = get_optional_extra(topic_model, "topics_over_time"),
    topics_per_class = get_optional_extra(topic_model, "topics_per_class"),
    reduced_embeddings_2d = get_optional_extra(topic_model, "reduced_embeddings_2d"),
    reduced_embeddings_3d = get_optional_extra(topic_model, "reduced_embeddings_3d")
  )
  saveRDS(extras, paste0(path, "_extras.rds"))
  invisible(TRUE)
}
