from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="decoder.calc_p",                # package.module path
        sources=["decoder/calc_p.pyx"],       # your pyx file
        include_dirs=[np.get_include()],     # for numpy arrays
    )
]

modules = cythonize(extensions)

setup(
    name="decoder",
    ext_modules=modules,
    include_dirs=[np.get_include()],
    py_modules=['behavior', 'calcium', 'decoder', 'divide_laps', 'dsp', 'field', 'model', 'multiday', 'simulator', 'stats']
)
