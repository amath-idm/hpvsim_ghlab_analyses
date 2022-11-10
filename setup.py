'''
Note: The scripts in this folder can be run directly; they do not need to be installed
via pip. This file is included largely for completeness.
'''

import os
import runpy
from setuptools import setup, find_packages

# Get version
cwd = os.path.abspath(os.path.dirname(__file__))
versionpath = os.path.join(cwd, 'version.py')
version = runpy.run_path(versionpath)['__version__']

# Deal with frozen requirements
with open('requirements_frozen.txt') as f: required_frozen = f.read().splitlines()

# Get the documentation
with open(os.path.join(cwd, 'README.md'), "r") as f:
    long_description = f.read()

CLASSIFIERS = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: Other/Proprietary License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Development Status :: 5 - Production/Stable",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]

setup(
    name="hpvsim_nairobi_scenarios",
    version=version,
    author="Robyn Stuart, Jamie Cohen, Cliff Kerr, Romesh Abeysuriya, Daniel Klein, Hao Hu",
    author_email="jamie.cohen@gatesfoundation.org",
    description="HPVsim Nairobi scenarios",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url='http://hpvsim.org',
    keywords=["HPV", "agent-based model", "simulation"],
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'hpvsim',
    ],
    extras_require={
        "frozen": required_frozen,
    }
)