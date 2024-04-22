General
=======
The Diffform python package implements differential forms and poly-forms from differential geometry. It includes some of the usual operations found in exterior calculus, include exterior product, differential operator. The main advatage of this package over other differential form packages ( e.g. [pycartan](https://github.com/TUD-RST/pycartan) ) is that it allows for polyforms and there is no dependence on basis forms. However, this removes some useful operations like insertion of vector fields.

This package is a part-time project during my PhD so updates should be suspected to end eventually. Bugs and mistakes may (possibly will) be prevalent.

Documentary will be implemented when I find the time, in the mean time I will try to provide comments in the code as a type of documentation.

ToDo List
=========
This is the list of possible implementation, in an approximate order of priority (interest to me):

- [X] Differential Forms
- [X] Exterior Product
- [X] Simplification of Forms
- [X] Exterior Differential Operator
- [ ] Substitution of factors/forms
- [ ] Vector fields
- [ ] Generic tensor product
- [ ] Insertion of vector fields
- [ ] Hodge star given metric 

Dependencies
============
Make sure you have the following python packages:

- wheel (needed for installing through pip)
- sympy

Installation
============
Package should be uploaded to pip fairly frequently and is currently under [diffforms](https://pypi.org/project/diffforms/)
