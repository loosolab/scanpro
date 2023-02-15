from setuptools import find_packages  # has to be imported befor distutils

from numpy.distutils.core import Extension, setup

flib = Extension(name='pypropeller.gaussq2',
                 extra_compile_args=['-O3'],
                 sources=['pypropeller/gaussq2.pyf', 'pypropeller/gaussq2.f'],  # you may add several modules files under the same extension
                 )

setup(
    name='pypropeller',
    version='0.2.0',
    packages=find_packages(include=['pypropeller', 'pypropeller.*']),
    install_requires=['pandas',
                      'statsmodels'],
    ext_modules=[flib]
)
