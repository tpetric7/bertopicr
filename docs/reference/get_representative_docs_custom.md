# Get Representative Documents for a Specific Topic

This function filters a given data frame to select a specified number of
representative documents from a particular topic. It uses random
sampling to select the documents.

## Usage

``` r
get_representative_docs_custom(df, topic_nr, n_docs)
```

## Arguments

- df:

  A data frame containing at least the columns 'Topic' and 'Document'.

- topic_nr:

  An integer specifying the topic number to filter the documents.

- n_docs:

  An integer specifying the number of documents to sample for the
  specified topic.

## Value

A vector of sampled documents corresponding to the specified topic.

## Examples

``` r
if (FALSE) { # \dontrun{
# Assuming `df_docs` is a data frame with columns `Topic`, `Document`, and `probs`
get_representative_docs_custom(df_docs, topic_nr = 3, n_docs = 5)
} # }
```
