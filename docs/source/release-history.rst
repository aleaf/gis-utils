===============
Release History
===============

Initial Release (2019-11-17)
----------------------------
* includes shp2df and df2shp functions for reading and writing .shp or .dbf files to/from pandas dataframes
* write_raster function for easy writing of GeoTiffs or Arc Ascii grids from numpy arrays
* get_values_at_points and zonal_stats functions for sampling rasters
* read_arc_ascii function for reading arc ascii grids with their metadata
* get_proj_str to get a PROJ string from a shapefile projection file
* project function for easy reprojections of shapely objects or x, y coordinates
