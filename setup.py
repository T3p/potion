from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))


setup(name='polyopt',
      packages=[package for package in find_packages()
                if package.startswith('polyopt')],
      install_requires=[],
      description="Policy optimization tools",
      author="Matteo Papini",
      url='',
      author_email="",
      version="0.1.1")
