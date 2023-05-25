from setuptools import find_packages  # has to be imported befor distutils

from numpy.distutils.core import Extension, setup

flib = Extension(name='scanpro.gaussq2',
                 extra_compile_args=['-O3'],
                 sources=['scanpro/gaussq2.pyf', 'scanpro/gaussq2.f'],  # you may add several modules files under the same extension
                 )

setup(
    name='scanpro',
    version='0.4.0',
    packages=find_packages(include=['scanpro', 'scanpro.*']),
    install_requires=['pandas',
                      'statsmodels',
                      'matplotlib',
                      'statannotations==0.4',  # statannotations doesn't support seaborn >= 0.12
                      # 'seaborn',
                      ],
    ext_modules=[flib])
