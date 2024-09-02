#' Set Up Python Environment for BERTopic
#'
#' This function sets up a Python environment with all required packages for using
#' the BERTopic model within the R package.
#' It checks if the specified environment exists and installs the necessary packages if needed.
#'
#' @param envname The name of the Python environment. Default is "r-bertopic".
#' @param python_path Optional path to a specific Python executable.
#' @return None
#' @export
setup_python_environment <- function(envname = "r-bertopic", python_path = NULL) {

  # Check if reticulate is available
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("The 'reticulate' package is required but is not installed. Please install it using install.packages('reticulate').")
  }

  # Initialize reticulate functions explicitly
  py_available <- reticulate::py_available
  use_python <- reticulate::use_python
  use_virtualenv <- reticulate::use_virtualenv
  py_module_available <- reticulate::py_module_available
  py_install <- reticulate::py_install

  # Check if Python is available
  if (!py_available(initialize = TRUE)) {
    stop("Python environment is not available. Please check your Python installation.")
  }

  # Specify Python environment path if provided
  if (!is.null(python_path)) {
    tryCatch({
      use_python(python_path, required = TRUE)
    }, error = function(e) {
      stop("Failed to use specified Python path: ", e$message)
    })
  } else {
    # Otherwise, use a virtual environment
    tryCatch({
      use_virtualenv(envname, required = FALSE)
    }, error = function(e) {
      stop("Failed to use or create the specified virtual environment: ", e$message)
    })
  }

  # Locate the requirements.txt file in the inst directory
  requirements_path <- system.file("requirements.txt", package = "bertopicr")

  if (file.exists(requirements_path)) {
    # Install packages listed in requirements.txt
    message("Installing Python packages from requirements.txt...")
    tryCatch({
      reticulate::py_install(
        packages = requirements_path,
        envname = envname,
        method = "auto",
        pip = TRUE  # Use pip to install from requirements.txt
      )
    }, error = function(e) {
      stop("Failed to install Python packages from requirements.txt: ", e$message)
    })
  } else {
    warning("requirements.txt file not found. Proceeding without installing additional packages.")
  }

  message("Python environment setup complete.")
}

# Example usage
# setup_python_environment(envname = "r-bertopic", python_path = "/usr/bin/python3")
