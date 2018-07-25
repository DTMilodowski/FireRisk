from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(['voxelise_modis_burned_area.pyx'],        # Cython code file
                          annotate=True),      # enables generation of the html annotation file
)
