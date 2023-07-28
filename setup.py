import re
from setuptools import find_packages  # has to be imported befor distutils
from numpy.distutils.core import Extension, setup


def find_version(file_path):
    version_file = open(file_path).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError(f"Unable to find version string in {file_path}.")


flib = Extension(name='scanpro.gaussq2',
                 extra_compile_args=['-O3'],
                 sources=['scanpro/gaussq2.pyf', 'scanpro/gaussq2.f'],  # you may add several modules files under the same extension
                 )

setup(
    name='scanpro',
    version=find_version('scanpro/_version.py'),
    description='Single-Cell Analysis of Proportions',
    author='Yousef Alayoubi',
    author_email='yousef.alayoubi@mpi-bn.mpg.de',
    license='MIT',
    packages=find_packages(include=['scanpro', 'scanpro.*']),
    install_requires=['pandas',
                      'statsmodels',
                      'matplotlib',
                      'statannotations>=0.4',  # statannotations doesn't support seaborn >= 0.12
                      'patsy',  # for creating design matrices
                      # 'seaborn',
                      ],
    ext_modules=[flib])
