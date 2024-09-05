# bertopicr

Topic modeling tools in `R` using the `reticulate` library as an interface to the `Python` package `BERTopic`.

## Introduction

The package `bertopicr` is based on the Python package `BERTopic` by *Maarten Grootendorst* (https://github.com/MaartenGr/BERTopic) and provides tools for performing unsupervised topic modeling. Topic modeling is a method for discovering the abstract "topics" that occur in a collection of documents. This package integrates `BERTopic` into `R` through the `reticulate` package, allowing seamless R-Python interoperability. It includes functions for visualization and analysis of topic modeling results, making it easier to explore topics within text data.

The `Python` package `BERTopic` is described in the paper: 

```bibtex
@article{grootendorst2022bertopic,
  title={BERTopic: Neural topic modeling with a class-based TF-IDF procedure},
  author={Grootendorst, Maarten},
  journal={arXiv preprint arXiv:2203.05794},
  year={2022}
}
```

## Installation

To install the package from `GitHub`, use the following command in `R`:

```r
devtools::install_github("tpetric7/bertopicr")
```

Ensure that you have the `devtools` package installed. If not, you can install it using:

```r
install.packages("devtools")
```

## Setting Up the Python Environment

This package requires a `Python` environment with specific packages to run `BERTopic` models. You can set up the environment using the following steps:

### Step 1: Install `bertopicr` using `devtools`

```r
devtools::install_github("tpetric7/bertopicr")
```

### Step 2: Set up the Python Environment

1. **Install Python**: Ensure Python is installed on your system. You can download it from [python.org](https://www.python.org/). To check if Python is installed, run:
   
    ```bash
    python --version
    ```

2. **Create a Virtual Environment**: It is recommended to create a virtual environment for the required Python packages:

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

    Clone (or download and unzip) the repository from GitHub:
    
    ```bash
    git clone https://github.com/tpetric7/bertopicr.git
    ```

    Change the working directory to the `inst` folder inside the cloned `bertopicr` repository:

    ```bash
    cd bertopicr/inst
    ```

    Use the `requirements.txt` file included in the package to install the necessary Python packages:

    ```bash
    pip install -r requirements.txt
    ```

    If your computer has a suitable GPU, it is recommended to install the `CUDA` version of `pytorch` in order to substantially accelerate processing. 
    For Windows (https://pytorch.org/get-started/locally/):
    
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

    Make sure to upgrade `pip` if necessary:

    ```bash
    python -m pip install --upgrade pip
    ```

5. **Running the Setup Function in R**:

    Alternatively, after installing `devtools` and the `bertopicr` package, run the setup function in `R` to install the `Python` dependencies:

    ```r
    library(reticulate)
    use_python("path/to/your/Python/env/r-bertopic")
    
    library(bertopicr)
    setup_python_environment()
    ```

    This function will set up the required Python environment and install all necessary packages.

### Step 6: Running Local Language Models

Download and install *ollama* (https://ollama.com/) or *lm-studio* (https://lmstudio.ai/). To install a language model, run the following command in a terminal (e.g., for `llama3.1`):

```bash
ollama pull llama3.1
```

If the ollama server does not start automatically, use:

```bash
ollama serve
```

In *lm-studio*, select a language model from the menu and start the server.

### Step 7: Install a _spaCy_ Language Model

On the spaCy website (https://spacy.io/models), choose a language model for your language. For Slovenian, you can install the following model:

```bash
python -m spacy download sl_core_news_md
```

## Handling Python Dependencies with `reticulate`

The `reticulate` package allows you to interface with Python from `R`. When using functions that rely on Python, you need to load Python modules dynamically within the R functions. For example:

```r
# Example function using reticulate to load BERTopic
#' Run BERTopic on Text Data
#'
#' This function runs BERTopic on a given set of text data and returns the topic model.
#' @param texts A character vector of text documents.
#' @return A BERTopic model object.
#' @export
run_bertopic <- function(texts) {
  library(reticulate)
  
  # Use your own Python environment
  use_python("path/to/your/python/env/r-bertopic", required = TRUE)
  reticulate::py_config()
  reticulate::py_available()

  # Import necessary Python modules
  bertopic <- import("bertopic")
  np <- import("numpy")
  sentence_transformers <- import("sentence_transformers")
  SentenceTransformer <- sentence_transformers$SentenceTransformer
  
  # Embeddings
  embedding_model <- SentenceTransformer("BAAI/bge-m3") # for multiple languages
  embeddings <- embedding_model$encode(texts, show_progress_bar = TRUE)
  
  # Initialize BERTopic model
  topic_model <- bertopic$BERTopic(embedding_model = embedding_model, calculate_probabilities = TRUE)
  
  # Fit the model on the text data
  fit_transform <- topic_model$fit_transform(texts, embeddings)
  topics <- fit_transform[[1]]
  probs <- fit_transform[[2]]
  
  return(list(topics, probs))
}
```

This example demonstrates how to use `reticulate` to load Python modules and perform topic modeling directly from `R`.

## Basic Usage

Once the package and Python environment are set up, you can use the following functions to perform topic modeling and visualize results:

```r
# Example usage
library(bertopicr)
library(dplyr)

# Load sample data
url <- "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-18/chocolate.csv"
chocolate <- readr::read_csv(url)

# Sample text data
texts <- chocolate$most_memorable_characteristics

# Run BERTopic on the text data
topic_model <- run_bertopic(texts)

# Analyze the topic results
topic_results <- tibble(Text = texts, Topic = topic_model[[1]], Probability = apply(topic_model[[2]], 1, max))

# Display the topics
topic_results
```

### Advanced Usage

#### Visualizing Topic Models

The package provides functions for visualizing topics, distributions, and hierarchical structures. Here are some examples:

```r
# Visualize topics
visualize_topics(topic_model)

# Visualize topic distribution
visualize_distribution(topic_model, text_id = 1, probabilities = probs)

# Visualize the hierarchical structure of topics
visualize_hierarchy(topic_model)
```

#### Custom Functions

You can use custom functions to extract specific information from your topic models. For example, to extract representative documents, to display the temporal development of topics or topic frequency within pre-defined classes or groups:

```r
# Get representative documents
representative_docs <- get_most_representative_docs(df_docs, topic_nr = 3, n_docs = 5)

# Visualize topics over time
visualize_topics_over_time(topic_model, topics_over_time, timestamps)

# Visualize topics per class
visualize_topics_per_class(topic_model, topics_per_class)
```

For model *training*, *dimension reduction* and *cluster selection*, run the enclosed quarto example file *topics_spiegel.qmd*. 

## Contributing

We welcome contributions! If you would like to contribute to this package, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Make your changes and test them.
4. Submit a pull request with a description of your changes.

Please ensure that your code follows the package's style and guidelines, and that you include tests for any new features.

## License

This package is licensed under the MIT License. You are free to use, modify, and distribute this software, provided that proper attribution is given to the original author.
