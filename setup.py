from setuptools import setup
from diffforms.release import __version__

with open('requirements.txt') as req_file:
    requirements = req_file.read()

setup(
    name='diffforms',
    version=__version__,
    url='https://github.com/Bone5505/Differential-Forms',
    author="Adam Shaw",
    packages=['diffforms'],
    install_requires=requirements,
    description="Symbolic differential forms python library",
    long_description=""" 
    A symbolic differential form library with exterior calculus operations 
    and sympy integration. Applications involve General Relativity and differential
    form descriptions of Manifolds.
    """,
    keywords="differential forms, sympy, polyforms, exterior derivative",
)