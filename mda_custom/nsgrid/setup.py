
# Ryan DeFever
# Sarupria Research Group
# Clemson University
# 2019 01 28

# Cython tutorial from 
# https://riptutorial.com/cython

from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy as np

ext = Extension(name="nsgrid_rsd", sources=["nsgrid_rsd.pyx"], language="c++",include_dirs=[np.get_include()])
#ext = Extension(name="tmp", sources=["tmp.pyx"], language="c++",include_dirs=[np.get_include()])
setup(ext_modules=cythonize(ext))
