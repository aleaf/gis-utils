===============
Release History
===============

Version 0.3.6 (2024-10-11)
--------------------------
* bug fix to get_values_at_points() to address breaking rasterio or gdal change where dataset.index() no longer accepts arrays for x and y

Version 0.3.5 (2023-07-19)
--------------------------
* bug fix (shp2df): replace ``DataFrame.append()`` call with ``pandas.concat`` for compatibility with ``pandas>=2.0``.

Version 0.3.4 (2023-02-03)
--------------------------
* bug fix (shp2df): fiona-related issue with checking for ``None`` type in a list containing geometry objects

Version 0.3.3 (2022-10-27)
--------------------------
* bug fix: stop using fiona-style CRS mappings for deprecated epsg and proj_str CRS input to df2shp (these are invalid inputs to the crs arg to fiona.open in newer versions)

Version 0.3.2 (2022-03-28)
--------------------------
* bug fix: replace use of osr with pyproj.crs.CRS to resolve windows DLL error
* bug fix: point pyproj to local environmental variables for SSL cert if pyproj network access fails

Version 0.3.1 (2022-02-08)
--------------------------
* this release is functionally the same as 0.3.0; was incremented due to issues with automated release workflow.

Version 0.3.0 (2022-02-08)
--------------------------
* drop support for python 3.8; add support for 3.10
* get_values_at_points: add support for NetCDF files
* project_raster: add kwargs
* remove project.py and import warning

Version 0.2.6 (2021-01-04)
--------------------------
* use WKT instead of proj dictionary mapping to define CRS when writing shapefiles (to avoid information loss; after https://pyproj4.github.io/pyproj/stable/crs_compatibility.html#converting-from-pyproj-crs-crs-for-fiona).

Version 0.2.5 (2020-10-20)
--------------------------
* fix raster.get_values_at_points to skip automatic reprojection if the raster is unprojected

Version 0.2.4 (2020-10-05)
--------------------------
* support for using pathlib Paths instead of strings
* :func:`~gisutils.raster.points_to_raster` method to interpolate point data to a regular grid and write to a GeoTIFF.
* fix raster.get_values_at_points to skip automatic reprojection if the raster is unprojected


Version 0.2.3 (2020-9-24)
--------------------------
* make rasterio optional dependency again, so that downstream programs that use the shapefile or projection modules
  don't need to depend on it


Version 0.2.2 (2020-9-9)
--------------------------
* add clip_raster() function to clip rasters to features, with automatic reprojection of features to the raster CRS
* replace `epsg` and `proj_str` arguments to :func:`df2shp` and :func:`write_raster`
  with general crs argument using `pyproj`

Version 0.2.0 (2020-8-26)
--------------------------
* add project_raster() function to reproject a raster to a different CRS
* optional points_crs argument to get_values_at_points() for automatic reprojection when sampling rasters
* optional destination CRS argument to shp2df for automatic reprojection when reading in shapefiles
* add get_authority_crs() function that returns a pyproj.crs.CRS instance for robust comparison of coordinate reference systems
* rename project module to projection, to avoid confusion with project() function

Version 0.1.4 (2020-5-14)
--------------------------
* fix bug in raster module where a negative sign was being added to the y-spacing by default

Version 0.1.3 (2020-4-12)
--------------------------
* added bilinear interpolation to raster.get_values_at_points

Version 0.1.2 (2020-2-12)
--------------------------
* fixed bug were project was returning shapely MultipartGeometry instances as lists

Version 0.1.1 (2020-6-12)
--------------------------
* made rasterio and rasterstats dependencies optional

Initial Release (2019-11-17)
----------------------------
* includes shp2df and df2shp functions for reading and writing .shp or .dbf files to/from pandas dataframes
* write_raster function for easy writing of GeoTiffs or Arc Ascii grids from numpy arrays
* get_values_at_points and zonal_stats functions for sampling rasters
* read_arc_ascii function for reading arc ascii grids with their metadata
* get_proj_str to get a PROJ string from a shapefile projection file
* project function for easy reprojections of shapely objects or x, y coordinates
