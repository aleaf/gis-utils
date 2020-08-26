gis-utils
-----------------------------------------------
Convenience functions for working with geospatial data in python. 



### Version 0.2
[![Build Status](https://travis-ci.com/aleaf/gis-utils.svg?branch=master)](https://travis-ci.com/aleaf/gis-utils)
[![Coverage Status](https://codecov.io/github/aleaf/gis-utils/coverage.svg?branch=master)](https://codecov.io/github/aleaf/gis-utils/coverage.svg?branch=master)
[![PyPI version](https://badge.fury.io/py/gis-utils.svg)](https://badge.fury.io/py/gis-utils)
[![Anaconda-Server Badge](https://anaconda.org/atleaf/gis-utils/badges/installer/conda.svg)](https://conda.anaconda.org/conda-forge)
[![Anaconda-Server Badge](https://anaconda.org/atleaf/gis-utils/badges/version.svg)](https://anaconda.org/conda-forge/gis-utils)



why gis-utils?
-------------------------
The python geospatial stack (`fiona`, `shapely`, `rasterio`, `pyproj`, `rtree`, and others) combined with `numpy` and `pandas` provides enormous power and flexibility for geospatial analysis. However, common tasks such as reading and writing shapefiles and rasters, reprojecting vector features, or rasterizing vector data typically require several lines of boiler-plate code that can be cumbersome and easy to forget. `gis-utils` aims to distill these operations into simple, robust functions that can be imported into any workflow, allowing effort to be focused on the more important parts of the analysis. 

For vector data, the [GeoPandas](http://geopandas.org/) project offers similar functionality in GeoDataFrame and GeoSeries structures that are tightly coupled to pandas. `gis-utils` aims to provide a looser, more minimal coupling, allowing the user more flexibility in the design of their workflows. For example, similar to GeoPandas, a shapefile can be read into a DataFrame with a single line of code, but instead of a GeoDataFrame, a standard DataFrame with a column of shapely objects is returned; it is
up to the user to develop their own workflows using standard python data structures. `gis-utils` also provides functions for working with rasters.


Getting Started
-----------------------------------------------


### Bugs

If you think you have discovered a bug in gis-utils in which you feel that the program does not work as intended, then we ask you to submit a [Github issue](https://github.com/aleaf/gis-utils/labels/bug).


Installation
-----------------------------------------------

**Python versions:**

gis-utils requires **Python** 3.6 (or higher)

**Dependencies:**  
numpy   
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
### Install in current location (to current python path)
(i.e., for development)  

```  
pip install -e .
```




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


