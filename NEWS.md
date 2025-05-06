# bertopicr 0.3.1

* Added `.onload()` function to allow `MacOS` users to import `BERTopic` and other installed `Python` modules from the virtual environment.
* If you update the `Python` package `BERTopic` to version `0.17.0`, you might have to update other packages as well or (like myself) to create a new virtual environment with compatible packages.
* After updating to `BERTopic=0.17.0`, you might experience that the `visualize_documents()` function doesn't render the dots in the scatterplot. A simple **temporary fix** is to open the `Python` file `_documents.py` of the `visualize_documents()` function of` BERTopic` (on my `Windows` system it sits in `anaconda3\envs\bertopic\Lib\site-packages\bertopic\plotting\`) and change `go.Scattergl` to `go.Scatter` in the `fig.add_trace()` function (it occurs twice in the `Python` script).

