# bertopicr

Topic modeling tools in R using the reticulate library as interface to the Python package BERTopic

## Introduction

This package provides tools for performing topic modeling using the BERTopic model, integrated into R through the `reticulate` package. It includes functions for visualization and analysis of topic modeling results, making it easier to understand and explore topics within text data.

## Installation

To install the package from GitHub, use the following command in R:

```r
devtools::install_github("tpetric7/bertopicr")
```

Ensure that you have the `devtools` package installed. If not, you can install it using:

```r
install.packages("devtools")
```

## Setting Up the Python Environment

This package requires a Python environment with specific packages to run BERTopic models. You can set up the environment using the following steps:

1. **Install Python**: Ensure Python is installed on your system. You can download it from [python.org](https://www.python.org/).

2. **Create a Virtual Environment**: It is recommended to create a virtual environment for the required Python packages.

    ```bash
    python -m venv r-bertopic
    ```

3. **Activate the Virtual Environment**:

    - On Windows:

        ```bash
        r-bertopic\Scripts\activate
        ```

    - On macOS and Linux:

        ```bash
        source r-bertopic/bin/activate
        ```

4. **Install Required Python Packages**:

    Use the `requirements.txt` file included in the package:

    ```bash
    pip install -r requirements.txt
    ```

5. **Running the Setup Function in R**:

    After installing the R package, run the setup function to ensure that all Python dependencies are properly installed:

    ```r
    library(bertopicr)
    setup_python_environment()
    ```

    This function will set up the required Python environment and install all necessary packages.

## Handling Python Dependencies with `reticulate`

Using the `reticulate` package, you can dynamically manage and interface with Python. When writing functions that require Python, ensure you explicitly load Python modules within the R functions, as shown below:

```r
# Example function using reticulate to load BERTopic
#' Run BERTopic on Text Data
#'
#' This function runs BERTopic on a given set of text data and returns the topic model.
#' @param texts A character vector of text documents.
#' @return A BERTopic model object.
#' @export
  use_python("c:/Users/teodo/anaconda3/envs/bertopic", required = TRUE)
  reticulate::py_config()
  reticulate::py_available()

  run_bertopic <- function(texts) {
  library(reticulate)
  use_python("c:/Users/teodo/anaconda3/envs/bertopic", required = TRUE)
  
  # Import necessary Python modules
  bertopic <- import("bertopic")
  BERTopic <- bertopic$BERTopic
  np <- import("numpy")
  sentence_transformers <- import("sentence_transformers")
  SentenceTransformer <- sentence_transformers$SentenceTransformer
  
  # Embeddings
  embedding_model = SentenceTransformer("BAAI/bge-m3")
  embeddings = embedding_model$encode(texts, show_progress_bar=TRUE)
  
  # Initialize BERTopic model
  topic_model <- BERTopic(embedding_model = embedding_model, 
                          calculate_probabilities = TRUE)
  
  # Fit the model on the text data
  fit_transform <- topic_model$fit_transform(texts, embeddings)
  topics <- fit_transform[[1]]
  probs <- fit_transform[[2]]
  
  return(c(topics, probs))
}
```

This example demonstrates how to use `reticulate` to load Python modules and perform topic modeling directly from R.

## Basic Usage

Once the package and Python environment are set up, you can use the following functions to perform topic modeling and visualize results:

```r
# Example usage
library(bertopicr)

url <- "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-18/chocolate.csv"
chocolate <- readr::read_csv(url)

# Sample text data
texts <- chocolate$most_memorable_characteristics

df <- data.frame(Text = texts)

# Run BERTopic on the text data
topic_model <- run_bertopic(texts)

topic_results <- df |> 
  mutate(Topic = topics, 
         Probability = apply(probs, 1, max))

# Display the resulting topic model
topic_results
```

This example shows how to use the `run_bertopic()` function to fit a BERTopic model to a set of text documents.

## Advanced Usage

### Visualizing Topic Models

The package also provides functions for visualizing topics over time, distributions, and hierarchical structures. Here are some examples:

```r
# Visualize topics over time
visualize_topics_over_time(topic_model, timestamps)

# Visualize topic distribution
visualize_distribution(topic_model)

# Visualize hierarchical structure of topics
visualize_hierarchy(topic_model)
```

### Custom Functions

You can create custom functions to extract specific information from your topic models. For example, extracting representative documents or visualizing topics per class:

```r
# Get representative documents
representative_docs <- get_most_representative_docs(df_docs, topic_nr = 3, n_docs = 5)

# Visualize topics per class
visualize_topics_per_class(topic_model, topics_per_class)
```

## Contributing

We welcome contributions! If you would like to contribute to this package, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Make your changes and test them.
4. Submit a pull request with a description of your changes.

Please ensure that your code follows the style and guidelines of the package, and that you include tests for any new features.

## License

This package is licensed under the MIT License. You are free to use, modify, and distribute this software, provided that proper attribution is given to the original author.
