===============
Release History
===============

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
