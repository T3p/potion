from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))


setup(name='potion',
      packages=[package for package in find_packages()
                if package.startswith('potion')],
      install_requires=['numpy',
              'scipy',
              'joblib',
              'torch==2.1',
              'tensorboardX',
              'matplotlib',
              'jupyter',
              'pandas',
              'gym'],
      description="Policy Optimization tools",
      author="Matteo Papini",
      url='https://github.com/T3p/potion',
      author_email="matteo.papini@polimi.it",
      version="0.1.1")
