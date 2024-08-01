import re
from setuptools import find_packages, setup  # has to be imported befor distutils


def find_version(file_path):
    version_file = open(file_path).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError(f"Unable to find version string in {file_path}.")


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
    setup_requires=['numpy<=1.26.4'],
    install_requires=['pandas',
                      'statsmodels',
                      'matplotlib',
                      'numpy<=1.26.4',  # included since numpy 2.0 produce error with pandas
                      'seaborn',
                      'patsy',  # for creating design matrices
                      ])
