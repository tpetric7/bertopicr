# Changelog

## bertopicr 0.3.6

- Documentation & Vignette fixes and bump version to 0.3.6

## bertopicr 0.3.5

- Added
  [`setup_python_environment()`](https://tpetric7.github.io/bertopicr/reference/setup_python_environment.md)
  helper for reticulate setup.
- Added
  [`train_bertopic_model()`](https://tpetric7.github.io/bertopicr/reference/train_bertopic_model.md)
  helper for end-to-end model training.
- Added
  [`save_bertopic_model()`](https://tpetric7.github.io/bertopicr/reference/save_bertopic_model.md)
  and
  [`load_bertopic_model()`](https://tpetric7.github.io/bertopicr/reference/load_bertopic_model.md)
  to persist and reload models with extras.
- Added vignettes `train_and_save_model.Rmd` and
  `load_and_reuse_model.Rmd`.
- Added demo script `inst/scripts/train_model_function_demo.R`.

## bertopicr 0.3.4

- The
  [`visualize_documents_2d()`](https://tpetric7.github.io/bertopicr/reference/visualize_documents_2d.md)
  function with richer labeling options was added to the already
  existing ones for document visualization
  ([`visualize_documents()`](https://tpetric7.github.io/bertopicr/reference/visualize_documents.md)
  and
  [`visualize_documents_3d()`](https://tpetric7.github.io/bertopicr/reference/visualize_documents_3d.md)).
- The
  [`visualize_documents_3d()`](https://tpetric7.github.io/bertopicr/reference/visualize_documents_3d.md)
  function also includes improved labeling options than in previous
  versions.

## bertopicr 0.3.3

- The update of the `Python` package `BERTopic` to version 0.17.4 and
  its dependencies introduced conflicts with certain `R` packages. Avoid
  attaching `R` libraries like `arrow` or `plotly` to the workflow.
- Due to the update of the `Python` package `BERTopic` and its
  dependencies, it is recommended to load the `Python` packages
  `BERTopic` and `sentence-transformers` before certain `R` libraries,
  e.g., `readr_rds()`.
- The `README.md` file was rewritten for easier understanding of the
  installation options and requirements.
- A `vignette` was added (alongside the `Quarto` tutorial and the
  pre-computed `HTML` file).
- Setup of a `pkgdown` website on `Github`
  (<https://tpetric7.github.io/bertopicr>) and another website on
  `Netlify` (<https://bertopicr.netlify.app/>).

## bertopicr 0.3.2

- If you update the `Python` package `BERTopic` to version `0.17.0`, you
  might have to update other packages as well or (like myself) to create
  a new virtual environment with compatible packages.
- After updating to `BERTopic=0.17.0`, you might experience that the
  [`visualize_documents()`](https://tpetric7.github.io/bertopicr/reference/visualize_documents.md)
  function doesn’t render the dots in the scatterplot. A simple
  **temporary fix** is to open the `Python` file `_documents.py` of the
  [`visualize_documents()`](https://tpetric7.github.io/bertopicr/reference/visualize_documents.md)
  function of`BERTopic` (on my `Windows` system it sits in
  `anaconda3\envs\bertopic\Lib\site-packages\bertopic\plotting\`) and
  change `go.Scattergl` to `go.Scatter` in the `fig.add_trace()`
  function (it occurs twice in the `Python` script).

## bertopicr 0.3.1

- Added `.onload()` function to allow `MacOS` users to import `BERTopic`
  and other installed `Python` modules from the virtual environment.

## bertopicr 0.3.0

- The `R` package `bertopicr` introduces training and visualization
  helpers to reproduce functions of the `Python` `BERTopic` package. For
  easier memorization, the functions names are (nearly) identical to
  those of the `Python` package `BERTopic`.

## bertopicr 0.2

- A preliminary collection of `R` wrapping functions of the `Python`
  `BERTopic` package and a tutorial for `R` users.
