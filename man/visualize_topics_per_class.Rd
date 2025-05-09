% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/visualize_topics_per_class.R
\name{visualize_topics_per_class}
\alias{visualize_topics_per_class}
\title{Visualize Topics per Class}
\usage{
visualize_topics_per_class(
  model = topic_model,
  topics_per_class = topics_per_class,
  start = 0,
  end = 10,
  filename = "topics_per_class",
  auto_open = TRUE
)
}
\arguments{
\item{model}{A BERTopic model object. Default is 'topic_model'.}

\item{topics_per_class}{A data frame or list containing the topics per class data. Default is 'topics_per_class'.}

\item{start}{An integer specifying the starting index of the topics to visualize. Default is 0.}

\item{end}{An integer specifying the ending index of the topics to visualize. Default is 10.}

\item{filename}{A string specifying the name of the HTML file to save the visualization. Default is "topics_per_class".}

\item{auto_open}{A logical value indicating whether to automatically open the HTML file after saving. Default is TRUE.}
}
\value{
A Plotly visualization of the topics per class, displayed as an HTML file within the R environment.
}
\description{
This function visualizes the distribution of topics per class using a pre-trained BERTopic model.
The visualization is generated using the Plotly Python package and displayed within an R environment.
}
\examples{
\dontrun{
visualize_topics_per_class(model = topic_model,
                           topics_per_class = topics_per_class,
                           start = 0, end = 7,
                           filename = "plot",
                           auto_open = TRUE)
}
}
