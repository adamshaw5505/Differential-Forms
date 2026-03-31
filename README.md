Archived
=======

The development for this project has moved to [codeberg](https://codeberg.org/adamshaw5505/Differential-Forms), this is a read-only copy made at the time of migration and will not recieve any more updates.


General
=======
The Diffform python package implements differential forms and poly-forms from differential geometry. It includes some usual operations found in exterior calculus, include exterior product, differential operator. The main advantage of this package over other differential form packages ( e.g. [pycartan](https://github.com/TUD-RST/pycartan) ) is that it allows for polyforms and there is no dependence on basis forms.

This package is a part-time project during my PhD, updates should be suspected to end eventually. Bugs and mistakes may (possibly will) be prevalent.

Documentation will be implemented when I find the time, when I have time I will try to implement comments in the class/functions as a rudimentary form of documentation.

ToDo List
=========
This is the list of possible implementation, in an approximate order of priority (interest to me):

- [X] Differential Forms
- [X] Exterior Product
- [X] Simplification of Forms
- [X] Exterior Differential Operator
- [X] Substitution of factors/forms
- [X] Vector fields
- [X] Generic tensor product
- [X] Insertion of vector fields
- [X] Generic Tensor Contractions
- [X] Implement substitution for Tensors
- [X] Hodge star given metric/frame (Working?)
- [X] Spin connection computation
- [ ] Lie Derivative on generic Tensor (add weight)
- [ ] Fallback to symbolic Hodge star without basis (or if form isn't in basis)
- [ ] Solving 1-form simple linear equations
- [ ] Vector Field Commutator
- [ ] Separate G-Structure Commands into separate files/subfolders

Dependencies
============
Make sure you have the following python packages:

- wheel (needed for installing through pip)
- sympy

Installation
============
Package should be uploaded to pip fairly frequently and is currently under [diffforms](https://pypi.org/project/diffforms/)
