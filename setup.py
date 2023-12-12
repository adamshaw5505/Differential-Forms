from setuptools import setup
from diffform.release import __version__

with open("requirements.py") as requirements_file:
    requirements = requirements_file.read()

setup(
    name='diffform',
    version=__version__,
    authoer="Adam Shaw",
    packages=['diffform'],
    description="General symbolic differential form python library",
    long_description=""" 
    A symbolic differential form library with exterior calculus operations 
    and sympy integration. Applications involve General Relativity and differential
    form descriptions of Manifolds.
    """,
    keywords="differential forms, sympy, polyforms, exterior derivative",
    install_equires=requirements,
)