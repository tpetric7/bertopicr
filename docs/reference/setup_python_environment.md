# Set Up Python Environment for BERTopic

This function sets up a Python environment with all required packages
for using the BERTopic model within the R package. It can create and
activate a virtualenv or conda environment and then install the bundled
requirements.

## Usage

``` r
setup_python_environment(
  envname = "r-bertopic",
  python_path = NULL,
  method = c("virtualenv", "conda"),
  python_version = NULL,
  upgrade = TRUE,
  extra_packages = NULL
)
```

## Arguments

- envname:

  The name of the Python environment. Default is "r-bertopic".

- python_path:

  Optional path to a specific Python executable (virtualenv only).

- method:

  Environment type to create and use. One of "virtualenv" or "conda".

- python_version:

  Optional Python version for conda (e.g. "3.10").

- upgrade:

  Logical. If TRUE, passes –upgrade to pip installs. Default is TRUE.

- extra_packages:

  Optional character vector of additional Python packages to install.

## Value

Invisibly returns the active Python configuration.
