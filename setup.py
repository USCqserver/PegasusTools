from setuptools import setup, Extension
#from distutils.core import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "pegasustools.pgq.cypqubit", ["pegasustools/pgq/cypqubit.pyx"],
        extra_compile_args=["-std=c++17"]
    ),
    Extension(
        "pegasustools.pgq.util", ["pegasustools/pgq/util.pyx"],
        extra_compile_args=["-std=c++17"]
    ),
    Extension(
        "pegasustools.util.qac", ["pegasustools/util/qac.pyx"],
        extra_compile_args=["-std=c++17"]
    )
]

setup(
    name="Pegasus Tools",
    ext_modules=cythonize(extensions, annotate=True),
)