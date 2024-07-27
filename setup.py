import re
from setuptools import find_packages, dist  # has to be imported before distutils
from distutils.command.sdist import sdist
# Test if numpy is installed
try:
    from skbuild import setup
except Exception:
    # Else, fetch numpy if needed
    dist.Distribution().fetch_build_eggs(['scikit-build'])
    from skbuild import setup


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
    cmdclass={'sdist': sdist},
    setup_requires=['scikit-build>=0.13'],
    )
