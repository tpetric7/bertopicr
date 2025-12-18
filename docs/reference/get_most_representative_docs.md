# Get Most Representative Documents for a Specific Topic

This function filters a given data frame to select the most
representative documents for a specified topic based on their
probability scores. The documents are sorted by relevance in descending
order, and the top n documents are returned.

## Usage

``` r
get_most_representative_docs(df, topic_nr, n_docs = 5)
```

## Arguments

- df:

  A data frame containing at least the columns 'Topic', 'Document', and
  'probs'.

- topic_nr:

  An integer specifying the topic number to filter the documents.

- n_docs:

  An integer specifying the number of top representative documents to
  return. Defaults to 5.

## Value

A vector of the most representative documents corresponding to the
specified topic. If the number of documents available is less than
`n_docs`, all available documents are returned.

## Examples

``` r
if (FALSE) { # \dontrun{
# Assuming `df_docs` is a data frame with columns `Topic`, `Document`, and `probs`
get_most_representative_docs(df_docs, topic_nr = 3, n_docs = 5)
} # }
```
