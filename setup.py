from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))


setup(name='potion',
      packages=[package for package in find_packages()
                if package.startswith('potion')],
      install_requires=[
              'gym==0.26.2',
              'gymnasium==0.29.1',
              'joblib==1.4.2',
              'matplotlib==3.9.0',
              'matplot2tikz==0.5.4',
              'numpy==1.26.4',
              'pandas==2.2.2',
              'scipy==1.13.1',
              'torch==2.3.0'],
      extras_require={
              'mujoco': ['gymnasium[mujoco]==0.29.1'],
      },
      description="Policy Optimization Framework and Algorithms",
      author="Matteo Papini",
      url='https://github.com/T3p/potion',
      author_email="matteo.papini@polimi.it",
      version="0.2.1")
