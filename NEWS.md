# bertopicr 0.3.4

* The `visualize_documents_2d()` function with richer labeling options was added to the already existing ones for document visualization (`visualize_documents()` and `visualize_documents_3d()`). 
* The `visualize_documents_3d()` function also includes improved labeling options than in previous versions. 

# bertopicr 0.3.3

* The update of the `Python` package `BERTopic` to version 0.17.4 and its dependencies introduced conflicts with certain `R` packages. Avoid attaching `R` libraries like `arrow` or `plotly` to the workflow.
* Due to the update of the `Python` package `BERTopic` and its dependencies, it is recommended to load the `Python` packages `BERTopic` and `sentence-transformers` before certain `R` libraries, e.g., `readr_rds()`.
* The `README.md` file was rewritten for easier understanding of the installation options and requirements.
* A `vignette` was added (alongside the `Quarto` tutorial and the pre-computed `HTML` file).
* Setup of a `pkgdown` website on `Github` (https://tpetric7.github.io/bertopicr) and another website on `Netlify` (https://bertopicr.netlify.app/).

# bertopicr 0.3.2

* If you update the `Python` package `BERTopic` to version `0.17.0`, you might have to update other packages as well or (like myself) to create a new virtual environment with compatible packages.
* After updating to `BERTopic=0.17.0`, you might experience that the `visualize_documents()` function doesn't render the dots in the scatterplot. A simple **temporary fix** is to open the `Python` file `_documents.py` of the `visualize_documents()` function of` BERTopic` (on my `Windows` system it sits in `anaconda3\envs\bertopic\Lib\site-packages\bertopic\plotting\`) and change `go.Scattergl` to `go.Scatter` in the `fig.add_trace()` function (it occurs twice in the `Python` script).

# bertopicr 0.3.1

* Added `.onload()` function to allow `MacOS` users to import `BERTopic` and other installed `Python` modules from the virtual environment.

# bertopicr 0.3.0

* The `R` package `bertopicr` became functional equivalent with many of the popular functions of the `Python` `BERTopic` package. For easier memorization, the functions names are (nearly) identical to those of the `Python` package `BERTopic`.

# bertopicr 0.1

* A preliminary collection of `R` wrapping functions of the `Python` `BERTopic` package and a tutorial for `R` users.
