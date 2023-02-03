===============
Troublehsooting
===============

Invalid coordinate transformations ('inf' values returned)
----------------------------------------------------------
Often these are caused by mis-specified coordinate reference systems (CRS); for example a shapefile with coordinates in UTM but a projection file that specifies a geographic CRS. They can also be caused by a network error, however.

SSL verification as the cause of invalid transformations
***********************************************************
**TL;DR: gis-utils attempts to resolve SSL errors if they are encountered** by telling pyproj to find an SSL certificate bundle in your environmental variables, including ``SSL_CERT_FILE``. **If you are still getting an SSL error**, consider 

    * submitting `an Issue <https://github.com/aleaf/gis-utils/issues>`_, 
    * or, if you try again off whatever network requires the special certificate (i.e., at home instead of the office), that may fix it.

If pyproj doesn't have the information it needs locally to make a transformation (i.e. especially for more obscure transformations), `it will try to use its own version of libcurl to reach out to the web for additional files <https://proj.org/usage/network.html>`_, which it will then cache locally so that it doesn't have to do this again. If you are on a network that requires SSL verification with a special certificate (i.e. the USGS Network), this will result in an SSL error unless the `libcurl within pyproj` is pointed to your certificate.

By default, Pyproj uses the `certifi package <https://pypi.org/project/certifi/>`_ certificate bundle, which is located within the certifi install in your ``site-packages`` folder for your python distribution (you can see where by importing certifi (``import certifi``) and then ``certifi.where()``). The certify bundle can be overridden in ``pyproj`` via the :func:`pyproj.network.set_ca_bundle_path` function. A path to a specific certificate can be passed to this function, but it is perhaps more convenient to simply argue ``False``, in which case pyproj will look at several system variables, including ``SSL_CERT_FILE``, for a certificate bundle. If you simply append your special certificate to a suitable existing bundle and then reference it in your environmental variables (e.g., in ``~/.bash_profile`` or ``~/.zshenv`` as ``export SSL_CERT_FILE=<path to certificate>``), then pyproj will find it with::

    pyproj.network.set_ca_bundle_path(False)

Note that this command only applies to the current python session `and` scope.

**USGS Users:** You can find more detailed instructions for including the DOI certificate in your bundle and referencing it for various applications `here <https://github.com/usgs/best-practices/blob/master/ssl/WorkingWithinSSLIntercept.md>`_.