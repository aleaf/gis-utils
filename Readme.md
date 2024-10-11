gis-utils
-----------------------------------------------
Convenience functions for working with geospatial data in python. The python geospatial stack (`fiona`, `shapely`, `rasterio`, `pyproj`, `rtree`, and others) combined with `numpy` and `pandas` provides enormous power and flexibility for geospatial analysis. However, common geoprocessing tasks often require several lines of boiler-plate code that can be tedious, repetitive and difficult to remember. `gis-utils` aims to distill these operations into simple, robust functions that can be imported into any workflow, to reduce cognitive load and allow effort to be focused on the more important parts of the analysis. 



### Version 0.3
![Tests](https://github.com/aleaf/gis-utils/workflows/Tests/badge.svg)
[![Coverage Status](https://codecov.io/github/aleaf/gis-utils/coverage.svg?branch=master)](https://codecov.io/github/aleaf/gis-utils/coverage.svg?branch=master)
[![PyPI version](https://badge.fury.io/py/gis-utils.svg)](https://badge.fury.io/py/gis-utils)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

Getting Started
-----------------------------------------------
### [Demo of gis-utils](https://github.com/aleaf/gis-utils/blob/develop/examples/gis-utils_demo.ipynb)


Installation
-----------------------------------------------

**Python versions:**

gis-utils requires **Python** 3.10 (or higher)

**Dependencies:**  
numpy   
scipy  
pandas  
fiona  
gdal  
rasterio  
rasterstats  
shapely  
rtree  
pyproj  

### Install python and dependency packages
Download and install the [Anaconda python distribution](https://www.anaconda.com/distribution/).
Open an Anaconda Command Prompt on Windows or a terminal window on OSX.
From the root folder for the package (that contains `requirements.yml`), install the above packages from `requirements.yml`.

```
conda env create -f requirements.yml
```
activate the environment:

```
conda activate gisutils
```

### Install to site_packages folder
```
python setup.py install
```
### Install the source code in-place
(for development or git users that want to easily *pull* the most recent code)  
* don't forget the `'.'`!

```  
pip install -e .
```

### Bugs

If you think you have discovered a bug in gis-utils in which you feel that the program does not work as intended, then we ask you to submit a [Github issue](https://github.com/aleaf/gis-utils/labels/bug).


Disclaimer
----------

This software is preliminary or provisional and is subject to revision. It is
being provided to meet the need for timely best science. The software has not
received final approval by the U.S. Geological Survey (USGS). No warranty,
expressed or implied, is made by the USGS or the U.S. Government as to the
functionality of the software and related material nor shall the fact of release
constitute any such warranty. The software is provided on the condition that
neither the USGS nor the U.S. Government shall be held liable for any damages
resulting from the authorized or unauthorized use of the software.


