% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/setup_python_environment.R
\name{setup_python_environment}
\alias{setup_python_environment}
\title{Set Up Python Environment for BERTopic}
\usage{
setup_python_environment(envname = "r-bertopic", python_path = NULL)
}
\arguments{
\item{envname}{The name of the Python environment. Default is "r-bertopic".}

\item{python_path}{Optional path to a specific Python executable.}
}
\value{
None
}
\description{
This function sets up a Python environment with all required packages for using
the BERTopic model within the R package.
It checks if the specified environment exists and installs the necessary packages if needed.
}
