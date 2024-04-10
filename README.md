General
=======
The Diffform python package implements differential forms and poly-forms from differential geometry. It includes some of the usual operations found in exterior calculus, include exterior product, differential operator. The main advatage of this package over other differential form packages ( e.g. [pycartan](https://github.com/TUD-RST/pycartan) ) is that it allows for polyforms and there is no dependence on basis forms. However, this removes some useful operations like insertion of vector fields (which is done using substitutions).

This package is a part-time project during my PhD so updates should be suspected to end eventually. Bugs and mistakes may (possibly will) be prevalent.

Documentary will be implemented when I find the time, in the mean time I will try to provide comments in the code as a type of documentation.

ToDo List
=========
This is the list of possible implementation, in an approximate order of priority (interest to me):

- [X] Differential Forms
- [X] Exterior Product
- [X] Simplification of Forms
- [X] Exterior Differential Operator
- [X] Substitution of factors/forms
- [ ] Sympy function integration
- [ ] Arbitrary sympy factors (up to user to keep track of type)
- [ ] Integration of forms (bounds/limits of polyforms? Inheret bounds?
- [ ] More?


Dependencies
============
Make sure you have the following python packages:

- sympy

Installation
============
Until I implement an installation method, the `core.py` file can be copied and imported to gain all functionality.
