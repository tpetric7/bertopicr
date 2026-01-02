#' Set Up Python Environment for BERTopic
#'
#' This function sets up a Python environment with all required packages for using
#' the BERTopic model within the R package. It can create and activate a virtualenv
#' or conda environment and then install the bundled requirements.
#'
#' @param envname The name of the Python environment. Default is "r-bertopic".
#' @param python_path Optional path to a specific Python executable (virtualenv only).
#' @param method Environment type to create and use. One of "virtualenv" or "conda".
#' @param python_version Optional Python version for conda (e.g. "3.10").
#' @param upgrade Logical. If TRUE, passes --upgrade to pip installs. Default is TRUE.
#' @param extra_packages Optional character vector of additional Python packages to install.
#' @return Invisibly returns the active Python configuration.
#' @export
setup_python_environment <- function(
  envname = "r-bertopic",
  python_path = NULL,
  method = c("virtualenv", "conda"),
  python_version = NULL,
  upgrade = TRUE,
  extra_packages = NULL
) {

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("The 'reticulate' package is required but is not installed. Please install it using install.packages('reticulate').")
  }

  method <- match.arg(method)

  if (method == "virtualenv" && is.null(python_path) && .Platform$OS.type == "windows") {
    detected_python <- Sys.which("python")
    if (nzchar(detected_python)) {
      python_path <- detected_python
    }
  }

  if (!is.null(python_path) && method == "conda") {
    warning("python_path is ignored when method = 'conda'.")
  }

  if (method == "virtualenv") {
    if (!reticulate::virtualenv_exists(envname)) {
      message("Creating virtualenv: ", envname)
      reticulate::virtualenv_create(envname, python = python_path)
    }
    reticulate::use_virtualenv(envname, required = TRUE)
  }

  if (method == "conda") {
    conda_envs <- tryCatch(reticulate::conda_list(), error = function(e) NULL)
    if (is.null(conda_envs)) {
      stop("Conda does not appear to be available. Install Miniconda/Anaconda or use method = 'virtualenv'.")
    }
    if (!envname %in% conda_envs$name) {
      message("Creating conda env: ", envname)
      reticulate::conda_create(envname, python = python_version)
    }
    reticulate::use_condaenv(envname, required = TRUE)
  }

  requirements_path <- system.file("requirements.txt", package = "bertopicr")
  if (file.exists(requirements_path)) {
    message("Installing Python packages from requirements.txt...")
    reticulate::py_install(
      packages = c("-r", requirements_path),
      envname = envname,
      method = "auto",
      pip = TRUE,
      pip_options = if (upgrade) "--upgrade" else NULL
    )
  } else {
    warning("requirements.txt file not found. Proceeding without installing additional packages.")
  }

  if (!is.null(extra_packages) && length(extra_packages) > 0) {
    message("Installing additional Python packages...")
    reticulate::py_install(
      packages = extra_packages,
      envname = envname,
      method = "auto",
      pip = TRUE,
      pip_options = if (upgrade) "--upgrade" else NULL
    )
  }

  message("Python environment setup complete.")
  invisible(suppressWarnings(reticulate::py_config()))
}
