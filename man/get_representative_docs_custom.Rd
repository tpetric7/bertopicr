% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/get_representative_docs_custom.R
\name{get_representative_docs_custom}
\alias{get_representative_docs_custom}
\title{Get Representative Documents for a Specific Topic}
\usage{
get_representative_docs_custom(df, topic_nr, n_docs)
}
\arguments{
\item{df}{A data frame containing at least the columns 'Topic' and 'Document'.}

\item{topic_nr}{An integer specifying the topic number to filter the documents.}

\item{n_docs}{An integer specifying the number of documents to sample for the specified topic.}
}
\value{
A vector of sampled documents corresponding to the specified topic.
}
\description{
This function filters a given data frame to select a specified number of representative documents
from a particular topic. It uses random sampling to select the documents.
}
\examples{
\dontrun{
# Assuming `df_docs` is a data frame with columns `Topic`, `Document`, and `probs`
get_representative_docs_custom(df_docs, topic_nr = 3, n_docs = 5)
}

}
