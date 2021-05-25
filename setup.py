from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "pgq", ["pegasustools/pgq/*.pyx"],
        extra_compile_args=["-std=c++17"]
    )
]

setup(
    name="Pegasus Tools",
    ext_modules=cythonize(extensions),
)
