## Citing paper
This is the open-sourced Python implementation of the TGGLinesPlus algorithm. details about the algirhtm is available in the  paper entitled
"TGGLinesPlus: A robust topological graph-guided computer vision algorithm for line detection from images" (arxiv version).


## Files
* `.gitignore`
<br> Globally ignored files by `git`
  
* `environment.yml`
<br> `conda` environment description of relevant dependencies
  
To recreate the `conda` environment we use in this repository, please  run:
```python
conda env create -f environment.yml
```

And to activate the environment:
```python
conda activate skeleton_graph
```

If you should ever want to export the full list of package dependencies from this `conda`
environment, you can run:
```python
conda env export --no-builds | grep -v "prefix" > environment.yml
```
* `--no-builds`: this is an attempt to make this miniconda environment work on across different operating systems by removing the build information for each package. To read more about this and how to fix any issues you run into if the above commands for installing the environment don't work for you, please see this [excellent post](https://johannesgiorgis.com/sharing-conda-environments-across-different-operating-systems/)
* `grep -v "prefix"`: this hides the `prefix` portion of the YAML file that lists the directory for where your miniconda environments are housed. This is likely to be different from one computer to the next, so it is not necessary, and hides full path information if you are working in data-sensitive environments.

## Folders
### `data`
* `mnist`: the original [MNIST dataset stored in CSV format](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv){:target="_blank"} and a zip file of the [Chinese MNIST](https://www.kaggle.com/datasets/fedesoriano/chinese-mnist-digit-recognizer){:target="_blank"} dataset
* `scikit-image` the [[page]](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.page) and [[retina]](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.retina) ndarrays
* `rs_imagery`: remote sensing image files
* `deepcrack`: image `11215-5.png` selected from the [[DeepCrack]](https://github.com/yhlleo/DeepCrack) dataset
* `mass_roads`: image `11278840_15.tif` selected from the [[Massachusetts Roads]](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset) dataset
* `contours`: [[city contours]](https://www.cabq.gov/gis/geographic-information-systems-data) from Albuquerque, NM; the original shapefile was overlaid onto a basemap of the area using QGIS, then exported as a PNG file

### `notebooks`
All Jupyter Notebook demos are located here, alongside all of the processing and plotting methods used in a folder called `utils/`.
