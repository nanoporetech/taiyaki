from glob import glob
import os
from setuptools import setup, find_packages
from setuptools.extension import Extension
import sys

import taiyaki

version = taiyaki.__version__


# Minimal list of requirements for the package to work once installed
# Used by pip
install_requires = [
    "h5py >= 2.2.1",
    "numpy >= 1.14.5",
    #  Biopython 1.76 last version to support Python 3.5
    "biopython >= 1.63, <=1.76",
    "Cython >= 0.25.2",
    "ont_fast5_api >= 1.2.0",
    "matplotlib >= 2.0.0",
    "pysam >= 0.15.0",
    #  Scipy 1.5.1 last version to support Python 3.5
    "scipy >= 1, <=1.5.1",
    "statsmodels >= 0.10.1",
    "torch >= 1.5"
]


#  Build extensions
try:
    import numpy as np
    from Cython.Build import cythonize
    extensions = cythonize([
        Extension(
            "taiyaki.squiggle_match.squiggle_match",
            [os.path.join("taiyaki/squiggle_match", "squiggle_match.pyx"),
             os.path.join("taiyaki/squiggle_match", "c_squiggle_match.c")],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-fopenmp", "-std=c99"],
            extra_link_args=["-fopenmp"]),
        Extension(
            "taiyaki.ctc.ctc",
            [os.path.join("taiyaki/ctc", "ctc.pyx"),
             os.path.join("taiyaki/ctc", "c_crf_flipflop.c"),
             os.path.join("taiyaki/ctc", "c_cat_mod_flipflop.c")],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-fopenmp", "-std=c11", "-mavx2"],
            extra_link_args=["-fopenmp"])])
except ImportError:
    extensions = []
    sys.stderr.write(
        "WARNING: Numpy and Cython are required to build taiyaki extensions\n")
    if any([cmd in sys.argv for cmd in
            ["install", "build", "build_clib", "build_ext", "bdist_wheel"]]):
        raise


setup(
    name='taiyaki',
    version=version,
    description='Neural network model training for Nanopore base calling',
    maintainer='Tim Massingham',
    maintainer_email='tim.massingham@nanoporetech.com',
    url='http://www.nanoporetech.com',
    long_description='Taiyaki is a library to support training and ' +
    'developing new base calling models for Oxford Nanopore ' +
    'Technologies sequencing platforms.',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],

    packages=find_packages(
        exclude=["*.test", "*.test.*", "test.*", "test", "bin"]),
    exclude_package_data={'': ['*.hdf', '*.c', '*.h']},
    ext_modules=extensions,
    setup_requires=["pytest-runner", "pytest-xdist"],
    tests_require=["parameterized", "pytest"],
    install_requires=install_requires,
    dependency_links=[],
    zip_safe=False,
    scripts=[x for x in glob('bin/*.py')],

)
