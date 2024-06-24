import re
from setuptools import find_packages, dist  # has to be imported befor distutils
from distutils.command.sdist import sdist


# Test if numpy is installed
try:
    from numpy.distutils.core import Extension, setup
except Exception:
    # Else, fetch numpy if needed
    dist.Distribution().fetch_build_eggs(['numpy'])
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


# Readme from git
def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='scanpro',
    version=find_version('scanpro/_version.py'),
    description='Single-Cell Analysis of Proportions',
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='Loosolab',
    author_email='yousef.alayoubi@mpi-bn.mpg.de',
    url='https://github.com/loosolab/scanpro',
    license='MIT',
    packages=find_packages(include=['scanpro', 'scanpro.*']),
    cmdclass={'sdist': sdist},
    setup_requires=['numpy==1.26.4'],
    install_requires=['pandas',
                      'statsmodels',
                      'matplotlib<3.9',
                      'seaborn',
                      'statannotations>=0.4',  # statannotations doesn't support seaborn >= 0.12
                      'patsy',  # for creating design matrices
                      ],
    ext_modules=[flib])
