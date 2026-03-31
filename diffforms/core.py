from sympy import *
from sympy.physics.units.quantities import Quantity
from IPython.display import Math
from sympy.combinatorics import Permutation
from itertools import permutations
from copy import deepcopy
import re
import numbers
import numpy as np
from math import factorial, prod

""" Global Settings:
        _PRINT_ARGUMENTS : Boolean - True if arguments are displayed in functions when printing, False otherwise. Default: False
     """
_PRINT_ARGUMENTS = False

# TODO:
# - Add functions that construct the Einstein tensor and the intermediate tensors needed along the way

def drange(n,d,repetition=True): return variations(range(n),d,repetition)

class Manifold():
    """Class: Manifold
    
    Keeps track of the data for a given manifold.
    
    Attributes:
        - label(String):                                           Arbitrary name for the manifold, too keep track if there are two similar manifold.
        - dimension(Integer):                                      A whole number, the dimension.
        - signature(Integer):                                      +1 or -1 depending if the there is 0 or 1 minus in the signature respectively.
        - coords(List[Symbols]):                                   List of symbols being the coordinates.
        - basis(List[DifferentialForm/DifferentialFormMul]):       List of basis 1-forms that make a basis of the cotangent space.
        - tetrads(List[DifferentialForm/DifferentialFormMul]):     List of tetrad 1-forms for the metric and frame.
        - frame_inv(List[DifferentialForm/DifferentialFormMul]): Inverse tetrads as a list of VectorFields.
        - metric(Tensor):                                          The metric on the manifold
        - metric_inv(Tensor):                                      Inverse of the metric
        - vectors(List[Tensor,VectorField]):                       List of Vectors/VectorFields that form a basis of the tangent space.
        - christoffel_symbols(Tensor):                             Christoffel symbols for the metric, def "get_christoffel_symbols" defines this.
    """
    def __init__(self, label : str, dimension : int, signature : int = 1):
        """Initialise the Manifold
        
        Arguments:
            - label(String):      Arbitrary name for the manifold.
            - dimension(Integer): The dimension.
            - signature(Integer): Signature of the manifold.
        """
        assert(dimension > 0)
        assert(len(signature) == dimension)
        assert(set(signature) == set([1]) or set(signature) == set([1,-1]) or set(signature) == set([-1]))
        self.label               = label
        self.dimension           = dimension
        self.signature           = signature
        self.signature_prod      = prod(signature)
        self.coords              = None
        self.basis               = None
        self.vectors             = None

        self.frame               = None
        self.frame_inv           = None
        self.metric              = None
        self.metric_inv          = None
        self.christoffel_symbols = None
        self.riemann_curvature   = None
        self.ricci_curvature     = None
        self.ricci_scalar        = None
        self.einstein_tensor     = None
        self.epsilon_tensor      = None
        self.volume              = None
        self.volume_form         = None
        self.spin_connection     = None
        self.spin_curvature      = None
    
    def __eq__(self,other : Manifold) -> bool:
        """Equates Manifolds by their label, dimension and signature."""
        if isinstance(other,Manifold):
            return (self.label == other.label) and (self.dimension == other.dimension) and (self.signature == other.signature)
        return False
    
    def __len__(self) -> int:
        """Returns the dimension of the Manifold."""
        return self.dimension

    def set_coordinates(self,coordinates:list) -> None:
        """Give coordinates to the Manifold.

        Also defines basis and vectors as d(c) and d/d(c).
        
        Arguments:
            - coordinates(List[Symbols]): List of symbols
        """
        assert(len(coordinates) == self.dimension)
        self.coords = coordinates
        self.basis = [DifferentialForm(self,c,0).d for c in coordinates]
        self.vectors = vectorfields(self,coordinates)
    
    def clear_variables(self) -> None:
        self.frame               = None
        self.frame_inv           = None
        self.metric              = None
        self.metric_inv          = None
        self.christoffel_symbols = None
        self.riemann_curvature   = None
        self.einstein_tensor     = None
        self.epsilon_tensor      = None
        self.volume              = None
        self.volume_form         = None
        self.spin_connection     = None
        self.spin_curvature      = None

    def set_frame(self,frame) -> None:
        """Sets the tetrad variable to a list of 1-forms. Also creates the metric and inverse metric.
        
        Arguments:
            - frame(List[DifferentialForm/DifferentialFormMul]): List of 1-forms
        """
        self.clear_variables()
        self.frame = frame

    def get_frame(self) -> list[DifferentialFormMul]:
        """Returns the list of frames"""
        if self.frame == None: raise(NotImplementedError,"Tetrads need to be provided by the user.")
        return self.frame

    def get_inverse_frame(self) -> list[Tensor]:
        """Return the list of inverse frames"""
        if self.vectors == None: raise(NotImplemented,"Coordinate Basis must be introduce")
        if self.frame == None: raise(NotImplemented,"Frame must be supplied by user")
        if self.frame_inv == None:
            frame_matrix = Matrix([[e.insert(v) for v in self.vectors] for e in self.frame])
            frame_matrix_inv = frame_matrix.inv().T
            self.frame_inv = [sum([frame_matrix_inv[I,u]*self.vectors[u] for u in range(self.dimension)]) for I in range(self.dimension)]
        return self.frame_inv
    
    def get_volume(self) -> Expr:
        """Return volume element"""
        if self.volume == None:
            volume = self.get_volume_form()
            for v in self.vectors:
                volume = volume.insert(v)
            self.volume = volume

        return self.volume

    def get_volume_form(self) -> DifferentialFormMul:
        """ Computes and returns the volume form """
        if self.volume_form == None:
            self.volume_form = prod(self.frame)
        return self.volume_form

    def get_basis(self) -> list[DifferentialFormMul]:
        """ Returns the Manifold 1-forms basis."""
        return self.basis

    def get_vectors(self) -> list[Tensor]:
        """Returns the Manifold VectorField basis."""
        return self.vectors

    def get_metric(self) -> Tensor:
        """Returns the Manifold metric."""
        if self.metric == None:
            frame_D = [e.to_tensor() for e in self.get_frame()]
            self.metric = sum([self.signature[I]*frame_D[I]*frame_D[I] for I in range(self.dimension)])
        return self.metric
    
    def get_inverse_metric(self) -> Tensor: 
        """Returns the inverse metric for the Manifold"""
        if self.metric_inv == None:
            frame_inv = self.get_inverse_frame()
            self.metric_inv = sum([self.signature[I]*frame_inv[I]*frame_inv[I] for I in range(self.dimension)])
        return self.metric_inv

    def get_christoffel_symbols(self) -> Tensor:
        """ Returns the Christoffel symbols for the metric, calculates the Christoffel symbols for the metric if need be.""" 
        if self.christoffel_symbols == None:
            T_DDD = PartialDerivative(self.get_metric())
            g_UU_T_DDD = (self.get_inverse_metric()*T_DDD)
            Gamma_UDD_1 = Contract(g_UU_T_DDD,(1,3))
            self.christoffel_symbols = ((Gamma_UDD_1 + PermuteIndices(Gamma_UDD_1,(0,2,1)) - Contract(g_UU_T_DDD,(1,2)))/Number(2)).simplify()
        return self.christoffel_symbols

    def get_spin_connection(self):
        """Computes the spin connection for a given frame in n-dimensions"""
        if self.spin_connection == None:
            wIJ_K_symbols = Array([[[(+1 if I < J else -1)*symbols(fr"\omega^{{{min(I,J)}{max(I,J)}}}_{{{K}}}") if I!=J else 0 for K in range(self.dimension)] for J in range(self.dimension)] for I in range(self.dimension)])
            η = diag(self.signature)
            wI_J_K_symbols = Array([[[sum([wIJ_K_symbols[I,L,K]*η[L,J] for L in range(self.dimension)]) for K in range(self.dimension)] for J in range(self.dimension)] for I in range(self.dimension)])
            spin_connection = [[sum([wI_J_K_symbols[I,J,K]*self.basis[K] for K in range(self.dimension)]) for J in range(self.dimension)] for I in range(self.dimension)]
            frame = self.get_frame()
            torsion_equations = [ExteriorDerivative(frame[I]) + sum([spin_connection[I][J]*frame[J] for J in range(self.dimension)]) for I in range(self.dimension)]
            all_symbols = []
            for I in range(self.dimension):
                for J in range(I+1,self.dimension):
                    for K in range(self.dimension):
                        all_symbols.append(wIJ_K_symbols[I,J,K])
            all_equations = []
            for eq in torsion_equations:
                for fact in eq.factors:
                    if fact != 0:
                        all_equations.append(fact)
            spin_comps_sol = solve(all_equations,all_symbols)
            wI_J_K_symbols = wI_J_K_symbols.subs(spin_comps_sol)
            self.spin_connection = [[sum([wI_J_K_symbols[I,J,K]*self.basis[K] for K in range(self.dimension)]) for J in range(self.dimension)] for I in range(self.dimension)]
        return self.spin_connection

        def get_spin_curvature(self) -> list[list[DifferentialFormMul]]:
            if self.spin_curvature == None:
                spin_connection = self.get_spin_connection()
                return [[ExteriorDerivative(spin_connection[I][J]) + sum([spin_connection[I][K]*spin_connection[K][J] for K in range(self.dimension)]) for J in range(self.dimension)] for I in range(self.dimension)]
            return self.spin_curvature

    def get_levi_civita_symbol(self) -> Tensor:
        """Return totally antisymmetric tensor with indices up"""
        if self.epsilon_tensor == None:
            self.epsilon_tensor = Tensor(self)
            for indices in permutations(list(range(self.dimension))):
                self.epsilon_tensor.comps_list.append([self.vectors[i] for i in indices])
                self.epsilon_tensor.factors.append(LeviCivita(*indices))
        return self.epsilon_tensor

    def get_riemann_curvature_tensor(self) -> Tensor :
        """Computes the Riemann Curvature Tensor from the Christoffel symbols"""
        if self.riemann_curvature == None:
            G_UDD = self.get_christoffel_symbols()
            dG_DUDD = PartialDerivative(G_UDD)
            R_UDDD = PermuteIndices(dG_DUDD,(1,3,0,2)) + PermuteIndices(Contract(G_UDD*G_UDD,(2,3)),(0,3,1,2))
            self.riemann_curvature = (R_UDDD - PermuteIndices(R_UDDD,(0,1,3,2)))
        return self.riemann_curvature

    def get_ricci_curvature(self) -> Tensor:
        if self.ricci_curvature == None:
            R_UDDD = self.get_riemann_curvature_tensor()
            self.ricci_curvature = Contract(R_UDDD,(0,2))
        return self.ricci_curvature

    def get_ricci_scalar(self) -> Expr:
        if self.ricci_scalar == None:
            g_UU = self.get_inverse_metric()
            R_DD = self.get_ricci_curvature()
            self.ricci_scalar = Contract(R_DD*g_UU,(0,2),(1,3))
        return self.ricci_scalar

    def get_einstein_tensor(self) -> Tensor:
        R_DD = self.get_ricci_curvature()
        R = self.get_ricci_scalar()
        g_DD = self.get_metric()
        return R_DD - Number(1,2)*R*g_DD

    def get_einstein_tensor(self) -> Tensor:
        if self.einstein_tensor == None:
            R_DD = self.get_ricci_curvature()
            R    = self.get_ricci_scalar()
            g_DD = self.get_metric()
            self.einstein_tensor = R_DD - Number(1,2)*g_DD*R
        return self.einstein_tensor

    def get_metric_determinant(self) -> Expr:
        """Returns the determinant of the metric.

        Returns:
            - Scalar (numpy expression) that is the determinant.
        """
        return self.signature_prod*self.get_volume()**2        
        
class VectorField():
    """Class VectorField

    Symbolic vector basis object representation.

    Attributes:
        - symbol(Symbol): The symbol with which the derivative of the vector field will be taken. 

    """
    def __init__(self,manifold : Manifold, symbol : Symbol):
        """ Returns the vector field on a given Manifold and given the symbol that constitutes the derivative.

        Arguments:
            - manifold(Manifold): The manifold the vector will be associated too.
            - symbol(Symbol): The symbolic symbol that the derivative is taken with respect too.
        
        Returns:
            - VectorField
        """
        self.symbol = symbol
        self.manifold = manifold
    
    def __eq__(self, other : VectorField) -> bool:
        """ Checks if two vectors are equal.
        """
        return (self.symbol == other.symbol) and (self.manifold == other.manifold)

    def __hash__(self):
        """ Generates a unique hash for each VectorField.
        """
        return hash((self.symbol,self.manfiold))

    def __mul__(self,other):
        """ Multiplies two VectorFields together using the tensor project. """
        return TensorProduct(self,other)
    
    def __rmul__(self,other): 
        """Right multiplication version of __mul__. """
        return TensorProduct(other,self)

    def __neg__(self):
        """Return the negative of a vector field as a Tensor. """
        ret = Tensor(self.manifold)
        ret.comps_list = [[self]]
        ret.factors = [-1]
        return ret
    
    def __sub__(self,other):
        """Returns the difference of two vectors fields as a Tensor. """
        return self + (-other)
    
    def __rsub__(self,other):
        """Right subtraction of __sub__. """
        return (-self) + other
    
    def __add__(self,other):
        """Add together the VectorField with Scalar/Tensor/VectorField/DifferentialForm. """
        ret = Tensor(self.manifold)
        ret.comps_list = [[self]]
        ret.factors = [1]
        if isinstance(self,(int,float,Expr)):
            if other != 0:
                ret.comps_list.append([1])
                ret.factors.append(other)
        elif isinstance(other,(VectorField,DifferentialForm)):
            ret.comps_list.append([other])
            ret.factors.append(1)
        elif isinstance(other,DifferentialFormMul):
            return self + other.to_tensor()
        elif isinstance(other,Tensor):
            ret.comps_list += other.comps_list
            ret.factors += other.factors
        else:
            raise NotImplementedError
        
        return ret
    
    def __radd__(self,other): 
        """Right addition of __add__. """
        return self+other
        
    def _repr_latex_(self):
        """Returns a latex string for the vector field. """
        return "$\\partial_{"+latex(self.symbol)+"}$"

    def __str__(self):
        """Returns the String of the symbol."""

        # For some reason latex(symbol) doesn't work here in my implementation, need to see why.
        return fr"\partial_{{{str(self.symbol)}}}"

    def __call__(self,func):
        """ Applies basis vector on function """
        if isinstance(func,(int,float)):
            return Number(0)
        elif isinstance(func,Expr):
            res = func.diff(self.symbol)
            return (res if isinstance(res,Expr) else Number(res))
        elif isinstance(func,DifferentialForm):
            res = func(self)
            return (res if isinstance(res,Expr) else Number(res))
        else:
            raise NotImplementedError("Basis Vector can only act on functions.")

    def conjugate(self):
        """Returns the complex conjugate of the VectorField. """
        return VectorField(self.manifold,conjugate(self.symbol))

    __repr__ = _repr_latex_
    _latex   = _repr_latex_
    _print   = _repr_latex_

class Tensor(): 
    """ Class Tensor

    Represents a poly-tensor, a poly-tensor is an arbitrary sum of any number of products of the tangent and cotangent space.
    
    For example, a Tensor could be "t = a + b + a*b" where "a" is a DifferentialForm and "b" is a VectorField.

    Attributes:
        - manifold(Manifold): The Manifold that the Tensor is defined on.
        - comps_list(List[VectorField/DifferentialForm]): List of lists that contain either a VectorField or DifferentialForm. Each sub list is a product and the top most list is the addition of the sublists.
        - factors(List[Integer/Float/Expr]): List of factors that appear in front the product of basis VectorField/DifferentialForm product.
    """
    def __init__(self, manifold:Manifold):
        """Returns and empty Tensor that, mostly used as a temporary storage for a new Tensor. 
        
        Arguments:
            - manifold(Manifold): The Manifold the Tensor is defined on.

        Returns:
            Empty Tensor with a Manifold.
        """
        self.manifold = manifold
        self.comps_list = []
        self.factors = []
    
    def __add__(self,other):
        """Adds a Int/Float/Expr/DifferentialForm/VectorField/Tensor with the Tensor field.
        
        Arguments:
            - other(Int/Float/Expr/DifferentialForm/VectorField/Tensor): Other Tensor/Vector/DifferentialForm/Scalar to add to the Tensor.

        Returns:
            Tensor Field
         """
        ret = Tensor(self.manifold)
        ret.comps_list += self.comps_list.copy()
        ret.factors += self.factors.copy()
        if isinstance(other,Tensor):
            ret.comps_list +=  (other.comps_list)
            ret.factors += other.factors
        elif isinstance(other,DifferentialForm):
            ret.comps_list += [[other]]
            ret.factors += [Number(1)]
        elif isinstance(other,VectorField):
            ret.comps_list += [[other]]
            ret.factors += [Number(1)]
        elif isinstance(other,DifferentialFormMul):
            return self + other.to_tensor()
        elif isinstance(other,(float,int)):
            if other != 0: ret = self + DifferentialForm(self.manifold,Rational(other),0)
        elif isinstance(other,Expr):
            ret = self + DifferentialForm(self.manifold,other,0)
        else:
            raise NotImplementedError
        ret._collect_comps()
        return ret
    
    def __radd__(self,other):
        """Right addition of __add__. """
        return self + other

    def __sub__(self,other):
        """Subtract Int/Float/Expr/DifferentialForm/VectorField/Tensor from Tensor. """
        return self + (-other)
    
    def __rsub__(self,other):
        """Right Subtraction of __sub__. """
        return other + (-self)

    def __neg__(self):
        """Return the negative of the Tensor. """
        ret = Tensor(self.manifold)
        ret.comps_list = self.comps_list.copy()
        ret.factors = [-f for f in self.factors]
        return ret

    def __mul__(self,other):
        """Return the tensor product of this Tensor with another object. """
        return TensorProduct(self,other)

    def __rmul__(self,other):
        """Right multiplication version of __mul__. """
        return TensorProduct(other,self)

    def __div__(self,other): 
        """ Divide the tensor by a Scalar. """
        if isinstance(other,(int,float)): other = Number(other)
        return TensorProduct(self,1/other)

    def __truediv__(self,other): 
        """True divide the tensor by a Scalar. """
        if isinstance(other,(int,float)): other = Number(other)
        return TensorProduct(self,1/other)

    def _repr_latex_(self):
        """Returns the LaTeX String related to the Tensor. """
        if not _PRINT_ARGUMENTS:
            latex_str = "$" + "+".join([ "(" + remove_latex_arguments(self.factors[i]) + ")" + r" \otimes ".join([str(f) for f in self.comps_list[i]]) for i in range(len(self.comps_list))])  + "$"
        else: 
            latex_str = "$" + "+".join([ "(" + latex(self.factors[i]) + ")" + r" \otimes ".join([latex(f) for f in self.comps_list[i]]) for i in range(len(self.comps_list))])  + "$"
        if latex_str == "$$":
            return "$0$"
        return latex_str
    
    def is_vectorfield(self) -> bool:
        """Check if a Tensor contains only VectorField components, and is valued in the Tangent space only. """
        for f in self.comps_list:
            if len(f) != 1 or not isinstance(f[0],VectorField):
                return False
        return True
    
    def get_weight(self) -> [int, int]:
        """ Returns the "weight" (weight is a number that ) """
        if len(self.factors) == 0: return (0,0)
        first_weight = tuple(map(lambda x: int(isinstance(x,VectorField))-int(isinstance(x,DifferentialForm)),self.comps_list[0]))
        for i in range(1,len(self.factors)):
            current_weight = tuple(map(lambda x: int(isinstance(x,VectorField))-int(isinstance(x,DifferentialForm)),self.comps_list[i]))
            if current_weight != first_weight: return (None)
        return first_weight

    def get_weights_list(self) -> list[int]:
        """ Returns the "weights" for each additive term """
        return [tuple(map(lambda x: int(isinstance(x,VectorField))-int(isinstance(x,DifferentialForm)),self.comps_list[i])) for i in range(len(self.factors))]

    def get_sub_tensor(self,index : int) -> Tensor:
        ret = Tensor(self.manifold)
        ret.factors = [self.factors[index]]
        ret.comps_list = [self.comps_list[index]]
        return ret

    def _collect_comps(self) -> None:
        new_comps_list = []
        new_factors = []
        
        # Collect terms with the same basis.
        for i in range(len(self.comps_list)):
            if self.comps_list[i] not in new_comps_list:
                new_comps_list.append(self.comps_list[i])
                new_factors.append(self.factors[i])
            else:
                j = new_comps_list.index(self.comps_list[i])
                new_factors[j] += self.factors[i]
        
        # Remove the terms with zero factors, zero basis elements or absorb and identity basis element.
        i = 0
        while  i < len(new_comps_list):
            if new_factors[i] == 0:
                del new_factors[i]
                del new_comps_list[i]
                continue
            new_comps_strings = [str(f) for f in new_comps_list[i]]
            if '0' in new_comps_strings:
                del new_comps_list[i]
                del new_factors[i]
                continue
            if len(new_comps_list[i]) > 1 and '1' in new_comps_strings:
                new_comps_list[i].pop(new_comps_strings.index('1'))
            i+=1

        self.comps_list = new_comps_list
        self.factors = new_factors

    def _eval_simplify(self, **kwargs):
        """Internal function for Sympy simplify call. """
        ret = Tensor(self.manifold)
        ret.comps_list = self.comps_list.copy()
        ret.factors = [simplify(f) for f in self.factors]
        ret._collect_comps()
        return ret

    def subs(self,target,sub=None,simp=False):
        """Substitute function that replaces components or basis elements with any Tensor components. 
        
        Arguments:
            - target(Scalar/VectorField/DifferentialForm): The object that is being replaced by the substition algorithm.
            - sub(Scalar/VectorField/DifferentialForm[Mul]/Tensor): The object that will replace the target.
            - simp(Boolean): Boolean that decides if to simplify factors of the result.
        
        Returns:
            Tensor tensor with target replaced.
        """
        ret = Tensor(self.manifold)
        ret.factors = self.factors.copy()
        ret.comps_list = self.comps_list.copy()

        if isinstance(target,(DifferentialForm,VectorField)):
            new_comps_list = []
            new_factors_list = []
            for I in range(len(self.comps_list)):
                if target in self.comps_list[I]:
                    J = ret.forms_list[I].index(target)
                    if isinstance(sub,(float,int,AtomicExpr,Expr,Number)):
                        new_comps_list +=[ret.comps_list[i][:J] + ret.comps_list[i][J+1:]]
                        new_factors_list.append(ret.factors[i]*sub/target.factors[0])
                    elif isinstance(sub,(DifferentialForm,VectorField)):
                        new_comps_list += [ret.comps_list[I][:J] + [sub] + ret.comps_list[I][J+1:]]
                        new_factors_list.append(ret.factors[I])
                    elif isinstance(sub,Tensor):
                        for K in range(len(sub.factors)):
                            s = sub.comps_list[K]
                            f = sub.factors[K]
                            new_comps_list +=[ret.comps_list[I][:J] + s + ret.comps_list[I][J+1:]]
                            new_factors_list.append(ret.factors[I]*f)
                    else:
                        raise NotImplementedError("Substitution must be a DifferentialForm, VectorFeild or Tensor.")
                else:
                    new_comps_list += [ret.comps_list[I]]
                    new_factors += [ret.factors[I]]
        elif isinstance(target,Tensor):
            if len(target.factors) > 1: raise NotImplementedError("Cannot replace more than 1 term at a time")
            new_comps_list = []
            new_factors_list = []
            for i in range(len(ret.comps_list)):
                match_index = -1
                for j in range(len(ret.comps_list[i])-len(target.comps_list[0])+1):
                    if ret.comps_list[i][j:j+len(target.comps_list[0])] == target.comps_list[0]:
                        match_index = j
                        break
                if match_index != -1:
                    if isinstance(sub,Tensor):
                        for k in range(len(sub.factors)):
                            s = sub.comps_list[k]
                            f = sub.factors[k]
                            new_comps_list += [ret.comps_list[i][:match_index] + s + ret.comps_list[i][match_index+len(target.comps_list[0]):]]
                            new_factors_list.append(ret.factors[i]*f/target.factors[0])
                    elif isinstance(sub,(DifferentialForm,Tensor)):
                        new_comps_list += [ret.comps_list[i][:match_index] + [sub] + ret.comps_list[i][match_index+len(target.comps_list[0]):]]
                        new_factors_list.append(ret.factors[i]/target.factors[0])
                    elif isinstance(sub,(float,int,AtomicExpr,Expr,Number)):
                        new_comps_list +=[ret.comps_list[i][:match_index] + ret.comps_list[i][match_index+len(target.comps_list[0]):]]
                        new_factors_list.append(ret.factors[i]*sub/target.factors[0])
                else:
                    new_comps_list += [ret.comps_list[i]]
                    new_factors_list.append(ret.factors[i])
            ret.factors = new_factors_list
            ret.comps_list = new_comps_list
        elif isinstance(target,dict):
            for key in target:
                ret = ret.subs(key,target[key],simp=False)
        elif sub != None:
            for i in range(len(self.factors)):
                ret.factors[i] = ret.factors[i].subs(target,sub)
        
        if simp: ret = ret.simplify()
        return ret
    
    def apply_func_to_factors(self, func, **kwargs):
        """ Applies functions to all the factors in the tensor """
        ret = Tensor(self.manifold)
        ret.factors = [func(f, **kwargs) for f in self.factors]
        ret.comps_list = self.comps_list.copy()
        ret._collect_comps()
        return ret
    
    def simplify(self, **kwargs) -> Tensor: return self.apply_func_to_factors(simplify, **kwargs)
    def factor(self, **kwargs) -> Tensor:   return self.apply_func_to_factors(factor,   **kwargs)
    def expand(self,**kwargs) -> Tensor:    return self.apply_func_to_factors(expand,   **kwargs)

    def conjugate(self, **args) -> Tensor:
        """ Return the complex conjugate of the Tensor. """
        ret = Tensor(self.manifold)
        ret.comps_list = [[f.conjugate() for  f in f_list] for f_list in self.comps_list]
        ret.factors = [conjugate(f) for f in self.factors]
        return ret

    def to_differentialform(self):
        """ Project a Tensor that is purely built from DifferentialForm's to a true DifferentialForm built with the WedgeProduct. """
        if set(self.get_weight()) != set([-1]): raise TypeError("Tensor cannot be projected to a differential form")
        ret = DifferentialFormMul(self.manifold)
        ret.factors = deepcopy(self.factors)
        ret.forms_list = deepcopy(self.comps_list)

        ret.remove_squares()
        ret.remove_above_top()
        ret.sort_form_sums()
        ret.collect_forms()

        if ret.factors == [] and ret.forms_list == []: 
            return Number(0)

        return ret/Number(factorial(ret.get_degree()))

    _sympystr = _repr_latex_
    __repr__  = _repr_latex_
    _latex    = _repr_latex_
    _print    = _repr_latex_

class DifferentialForm():
    """
    Class: Differential Form

    This is the "atom" of a differential form. Holds a 1-form with 1 term.

    Attributes:
        - manifold(Manifold): Manifold the differential form exists on.
        - degree(Integer):    Degree of the differential form.
        - symbol(Symbol):     The Sympy symbol that represents the form.
        - exact(Boolean):     True/False depending on if the differential form is exact or not.    
    """

    def __init__(self,manifold : Manifold, symbol : Symbol, degree : int = 0, exact : bool = False):
        """Intialise the Differential form
        
        Arguments:
            - manifold(Manifold): The Manifold for the differential form.
            - symbol(Symbol):     The symbol to represent the differential form.
            - degree(Integer):    The degree of the form, must be greater than 0 and less than or equal to the dimension of the manifold.
            - exact(Boolean):     True if the form is closed and the exterior derivative is automatically zero. False otherwise.
        
        Returns:
            DifferentialForm represented bt the symbol.
         """
        self.manifold = manifold
        self.degree = degree
        self.symbol = symbol
        self.exact = exact
        if degree < 0 or degree > self.manifold.dimension:
            self.symbol = Rational(0)
        
    def __eq__(self,other : DifferentialForm) -> bool:
        """ Compares if two differential forms are equal. """
        if not isinstance(other,DifferentialForm): return False
        return (self.symbol == other.symbol) and (self.get_degree() == other.get_degree())
    
    def __hash__(self) -> int: 
        """ Unique hash for a differential form. """
        return hash((str(self.symbol),self.get_degree()))

    def __mul__(self,other) -> DifferentialFormMul: 
        """ Multiplies the DifferentialForm with a Tensor/VectorField to produce a Tensor, or another DifferentialForm to produce a DifferentialFormMul. """
        if isinstance(other,(Tensor,VectorField)):
            return TensorProduct(self,other)
        return WedgeProduct(self,other)

    def __rmul__(self,other) -> DifferentialFormMul: 
        """Right multiplication of __mul__. """
        if isinstance(other,(Tensor,VectorField)):
            return TensorProduct(other,self)
        return WedgeProduct(other,self)
    
    def __div__(self,other) -> DifferentialFormMul:
        """Divide the DifferentialForm by another object. """
        if isinstance(other,(int,float)): other = Number(other)
        return WedgeProduct(self,1/other)
    
    def __truediv__(self,other) -> DifferentialFormMul:
        """Truediv version of __div__. """
        if isinstance(other,(int,float)): other = Number(other)
        return WedgeProduct(self,1/other)

    def __add__(self,other) -> DifferentialFormMul:
        """Add together DifferentialForm and another object. """
        ret = DifferentialFormMul(self.manifold)
        ret.forms_list = [[self]]
        ret.factors = [1]
        if isinstance(other,(Expr,int,float)):
            if other != 0:
                ret.forms_list.append([])
                ret.factors.append(other)
        elif isinstance(other,DifferentialForm):
            ret.forms_list.append([other])
            ret.factors.append(1)
        elif isinstance(other,DifferentialFormMul):
            ret.forms_list += other.forms_list[:]
            ret.factors += other.factors[:]
        else:
            raise NotImplementedError
        ret.collect_forms()
        return ret
    
    def __radd__(self,other) -> DifferentialFormMul: 
        """Right addition version of __add__. """
        return self + other

    def __lt__(self,other) -> bool:
        """Less that operator to order differential forms. Ordered alphabetically by the String of the symbol. """
        if not isinstance(other,DifferentialForm): raise NotImplementedError
        if str(self.symbol) < str(other.symbol):
            return True
        elif str(self.symbol) > str(other.symbol):
            return False
        else:
            return (self.get_degree()) < other.get_degree()

    def __neg__(self) -> DifferentialFormMul:
        """Return the negative of a Differential Form. """
        return DifferentialFormMul(self.manifold,self,-1)
    
    def __sub__(self,other) -> DifferentialFormMul:
        """Subtract an object from the DifferentialForm. """
        return self + (-other)
    
    def __rsub__(self,other)  -> DifferentialFormMul: 
        """Right subtraction version of __sub__. """
        return (-self) + other

    def __str__(self) -> str:
        """Returns the LaTeX String of the symbol. """
        return latex(self.symbol)

    def _repr_latex_(self) -> str:
        """Sympy internal call that returns the LaTeX string of the symbol. """
        return self.symbol._repr_latex_()

    def __hash__(self) -> int:
        """Unique hash for a DifferentialForm. """
        return hash((self.symbol,self.get_degree()))

    def __call__(self,vect) -> DifferentialFormMul:
        return self.insert(vect)

    def to_tensor(self) -> Tensor:
        """Converts a DifferentialForm to a Tensor, such that multiplication uses the TensorProduct instead of WedgeProduct. """
        return (Number(1)*self).to_tensor()
    
    __repr__ = _repr_latex_
    _latex   = _repr_latex_
    _print   = _repr_latex_
    
    def __eq__(self,other) -> bool:
        """Tests if two DifferentialForms are equivalent. """
        if isinstance(other,DifferentialForm):
            return str(self.symbol) == str(other.symbol) and self.get_degree() == other.get_degree()
        return False

    def _eval_simplify(self, **kwargs):
        """Overrides sympy internal simplify call to return self. This object is already simplified by construction. """
        return self
    
    def insert(self, vector:VectorField) -> DifferentialFormMul | Expr | int | float:
        """ Insert a VectorField into the DifferentialForm. 
        
        Arguments:
            - vector(VectorField): The vector field that will be inserted into the DifferentialForm.
        
        Returns:
            Contraction of the DifferentialForm and VectorField as a Scalar.
        """
        if isinstance(vector,VectorField):
            if self.symbol == vector.symbol or str(self.symbol) == "d\\left("+str(vector.symbol)+"\\right)": return Number(1)
            else: return Number(0)
        elif isinstance(vector,Tensor):
            if vector.is_vectorfield():
                return sum([vector.factors[i]*self.insert(vector.comps_list[i][0]) for i in range(len(vector.factors))])
        else:
            raise NotImplementedError

    @property
    def d(self) -> DifferentialFormMul:
        """ Exterior derivative of the differential form, in the given manifold. 
        
        Computes the Exterior Derivative of a differential form. If the differential field is purely symbol it returns zero if self.exact=True.
        Allows for purely symbolic differential forms by return a new differental form with symbol = "d(old_symbol)" which is exact (closed).

        Returns:
            DifferntialFormMul

        """

        if self.exact: return Number(0)
        elif isinstance(self.symbol,Number): return Number(0)
        else:
            dsymbol = symbols(r"d\left("+str(self.symbol)+r"\right)",**self.symbol.assumptions0)
            return DifferentialForm(self.manifold,dsymbol,degree=self.get_degree()+1,exact=True)
        raise NotImplementedError

    def subs(self,target,sub=None) -> DifferentialFormMul:
        """Substitute function that replaces a differential with another DifferentialForm components.
        
        Arguments:
            - target(DifferentialFormMul): The object that is being replaced by the substition algorithm.
            - sub(Scalar/DifferentialForm): The object that will replace the target.
        
        Returns:
            DifferentialForm with target replaced.
        """
        if target == self: return sub
        elif isinstance(target,DifferentialFormMul):
            if len(target.factors) == 1 and target.forms_list == [[self]]:
                return sub/target.factors[0]
        elif isinstance(target,dict):
            ret = DifferentialForm(self.symbol,self.get_degree())
            ret.exact = self.exact
            for t in target:
                ret = ret.subs(t,target[t])
            return ret
        else:
            ret = DifferentialForm(self.symbol,self.get_degree())
            ret.exact = self.exact
            return ret

    def conjugate(self) -> DifferentialForm:
        """Return the complex conjugate of a DifferentialForm. """
        return DifferentialForm(self.manifold,conjugate(self.symbol),self.get_degree(),self.exact)

    def get_degree(self) -> int: 
        return self.degree

class DifferentialFormMul():
    """ Class: DifferentialFormMul

    Contains sums of products of the DifferentialForm class as a basis.

    Attributes:
        - forms_list(List[List[DifferentialForm]]): List of lists where the sub lists contaion the wedge product of DifferentialForms and the outer list represents the sum.
        - factors(List[Scalar]):                    List of factors for each term in the outer list of forms_list.
    """

    def __init__(self, manifold : Manifold, form : DifferentialForm = None, factor : Expr = None):
        """Initialise the DifferentialFormMul class, mostly used as a empty differential form to modify.
        
        Arguments:
            - manifold(Manifold):     Manifold on which the differential form is defined.
            - form(DifferentialForm): Used to create a differential form with 1 term.
            - factor(AtomicExpr):     Factor used to create differential form with 1 term.

        Returns:
            Empty or 1 term DifferentialForm.
         """
        if form == None:
            self.forms_list = []
            self.factors = []  
        else:
            self.forms_list = [[form]]
            self.factors = [factor]
        self.manifold = manifold
 
    def __add__(self, other : Expr | int | float | DifferentialForm | DifferentialFormMul) -> DifferentialFormMul:
        """ Adds another object to a differential form. """
        ret = DifferentialFormMul(self.manifold)
        ret.forms_list = self.forms_list.copy()
        ret.factors = self.factors.copy()
        if isinstance(other,DifferentialFormMul):
            assert(self.manifold == other.manifold)
            ret.forms_list += other.forms_list[:]
            ret.factors += other.factors[:]
        elif isinstance(other,DifferentialForm):
            assert(self.manifold == other.manifold)
            ret.forms_list.append([other])
            ret.factors.append(1)
        elif isinstance(other,(float,int,AtomicExpr,Number)):
            if other != 0:
                ret.forms_list.append([])
                ret.factors.append(other)
        else:
            raise NotImplementedError
        ret.remove_squares()
        ret.remove_above_top()
        ret.sort_form_sums()
        ret.collect_forms()

        if ret.factors == [] and ret.forms_list == []: return Number(0)
        elif ret.forms_list == [[]]: return ret.factors[0]
        return ret
    
    def __mul__(self, other) -> DifferentialFormMul: 
        """Multiply differential form with form/vectorfield or scalar. """
        if isinstance(other,(Tensor,VectorField)):
            return TensorProduct(self,other)
        return WedgeProduct(self,other)
    
    def __rmul__(self,other) -> DifferentialFormMul: 
        """Right multiplication version of __mul__. """
        if isinstance(other,(Tensor,VectorField)):
            return TensorProduct(other,self)
        return WedgeProduct(other,self)

    def __div__(self,other) -> DifferentialFormMul: 
        if isinstance(other,(int,float)): other = Number(other)
        return WedgeProduct(self,1/other)

    def __truediv__(self,other) -> DifferentialFormMul: 
        if isinstance(other,(int,float)): other = Number(other)
        return WedgeProduct(self,1/other)

    def __radd__(self,other) -> DifferentialFormMul: 
        return self + other
    
    def __sub__(self,other) -> DifferentialFormMul: 
        return self + (-other)
        
    def __rsub__(self,other) -> DifferentialFormMul: 
        return other + (-self)

    def __eq__(self,other) -> bool:
        """Checks if two differential forms are equivalent. """
        if isinstance(other,DifferentialForm) and self.factors == [1] and len(self.forms_list[0]) == 1: return other == self.forms_list[0][0]
        elif not isinstance(other,DifferentialFormMul): return False
        elif other.factors != self.factors: return False
        elif other.forms_list != self.forms_list: return False
        return True

    def __hash__(self) -> int: 
        """Unique hash for differential forms. """
        symbols = []
        for forms in self.forms_list: symbols+=forms
        symbols += self.factors
        return hash(tuple(symbols))

    def __neg__(self) -> DifferentialFormMul:
        ret = DifferentialFormMul(self.manifold)
        ret.forms_list = self.forms_list.copy()
        ret.factors = [-f for f in self.factors]
        return ret

    def insert(self, other : Tensor) -> DifferentialFormMul:
        """Insert a VectorField into a differential form. """
        if isinstance(other,VectorField):
            ret = DifferentialFormMul(self.manifold)
            for i in range(len(self.forms_list)):
                sign = 1
                for j in range(len(self.forms_list[i])):
                    if self.forms_list[i][j].insert(other) != 0:
                        ret.forms_list += [self.forms_list[i][:j] + self.forms_list[i][j+1:]]
                        ret.factors += [self.factors[i]*sign]
                        break
                    sign *= (-1)**self.forms_list[i][j].get_degree() 
        elif isinstance(other,Tensor) and other.is_vectorfield():
            ret = sum([other.factors[i]*self.insert(other.comps_list[i][0]) for i in range(len(other.factors))])
            return ret
        else:
            raise NotImplementedError("Tensor inserted must be a vector field")

        if ret.forms_list == [[]]: return ret.factors[0]
        if ret.forms_list == []: return Number(0)

        ret.remove_squares()
        ret.remove_above_top()
        ret.sort_form_sums()
        ret.collect_forms()
        return ret

    def remove_squares(self) -> None:
        """Removes the square of a 1-form. """
        i = 0
        while i < len(self.forms_list):
            deled = False
            for j in range(len(self.forms_list[i])):
                f = self.forms_list[i][j]
                if f.get_degree()%2 == 1 and self.forms_list[i].count(f) > 1:
                    del self.forms_list[i]
                    del self.factors[i]
                    deled = True
                    break
            if not deled: i+=1
        
    def remove_above_top(self) -> None:
        """Removes any differential form with degree above the top form. """
        i = 0
        while i < len(self.forms_list):
            if sum([f.get_degree() for f in self.forms_list[i]]) > self.manifold.dimension:
                del self.forms_list[i]
                del self.factors[i]
                continue
            i += 1

    def sort_form_sums(self) -> None:
        """Order the form product in consitent order. """
        for i in range(len(self.forms_list)):
            bubble_factor = 1
            for j in range(len(self.forms_list[i])):
                for k in range(j,len(self.forms_list[i])):
                    if self.forms_list[i][j] > self.forms_list[i][k]:
                        temp = self.forms_list[i][j]
                        self.forms_list[i][j] = self.forms_list[i][k]
                        self.forms_list[i][k] = temp
                        bubble_factor *= (-1)**(self.forms_list[i][j].get_degree()*self.forms_list[i][k].get_degree())
            self.factors[i] = self.factors[i]*bubble_factor
    
    def collect_forms(self) -> None:
        """Collect terms that have the same basis. Also remove terms that are zero after insertion or collapse indentity term. """
        new_forms_list = []
        new_factors = []
        for i in range(len(self.forms_list)):
            if self.forms_list[i] not in new_forms_list:
                new_forms_list.append(self.forms_list[i])
                new_factors.append(self.factors[i])
            else:
                j = new_forms_list.index(self.forms_list[i])
                new_factors[j] += self.factors[i]
        
        i = 0
        while  i < len(new_forms_list):
            if new_factors[i] == 0:
                del new_factors[i]
                del new_forms_list[i]
                continue
            i+=1
    
        i = 0
        while i < len(new_forms_list):
            new_forms_strings = [str(f) for f in new_forms_list[i]]
            if '0' in new_forms_strings:
                del new_forms_list[i]
                del new_factors[i]
                continue
            if len(new_forms_list[i]) > 1 and '1' in new_forms_strings:
                new_forms_list[i].pop(new_forms_strings.index('1'))
            i+=1

        self.forms_list = new_forms_list
        self.factors = new_factors
            
    def _repr_latex_(self) -> str:
        """Return the LaTeX String for a differential form. """
        if not _PRINT_ARGUMENTS:
            latex_str = "$" + "+".join([ "(" + remove_latex_arguments(self.factors[i]) + ")" + r" \wedge ".join([str(f) for f in self.forms_list[i]]) for i in range(len(self.forms_list))]) + "$"
        else:
            latex_str = "$" + "+".join([ "(" + latex(self.factors[i]) + ")" + r" \wedge ".join([str(f) for f in self.forms_list[i]]) for i in range(len(self.forms_list))]) + "$"
        if latex_str == "$$":
            return "$0$"
        return latex_str

    def get_degree(self) -> int:
        degree_set = set([sum([ssl.get_degree() for ssl in sl]) for sl in self.forms_list])
        if len(degree_set) == 1:
            return list(degree_set)[0]
        return None
    
    def __is_number(self):
        if self.forms_list == []:
            if len(self.factors) == 1: 
                if isinstance(self.factors[0],(int,float)): return Number(self.factors[0])
                return self.factors[0]
            else:
                return Number(0)
        return None
    
    def __call__(self,vect):
        return self.insert(vect)
    
    _sympystr = _repr_latex_
    __str__ = _repr_latex_

    @property
    def d(self) -> DifferentialFormMul:
        """Take the Exterior derivative of a differential form. """
        ret = DifferentialFormMul(self.manifold)
        new_forms_list = []
        new_factors_list = []
        for i in range(len(self.forms_list)):
            fact = self.factors[i]
            if hasattr(fact,"free_symbols"):
                for f in fact.free_symbols:
                    dfact = fact.diff(f)
                    if dfact != 0:
                        new_forms_list += [[DifferentialForm(self.manifold,f,0).d] + self.forms_list[i]]
                        new_factors_list += [dfact]
            for j in range(len(self.forms_list[i])):
                d_factor = (-1)**sum([0] + [f.get_degree() for f in self.forms_list[i][0:j]])
                dform = self.forms_list[i][j].d
                if dform == 0: continue
                new_forms_list += [self.forms_list[i][0:j] + [dform] + self.forms_list[i][j+1:]]
                new_factors_list += [d_factor*self.factors[i]]

        ret.forms_list = new_forms_list
        ret.factors = new_factors_list

        ret.remove_squares()
        ret.remove_above_top()
        ret.sort_form_sums()
        ret.collect_forms()

        r = ret.__is_number()

        if r != None: return r

        return ret

    def _eval_simplify(self, **kwargs) -> DifferentialFormMul:
        """Override sympy internal simplify call for differential form. """
        ret = DifferentialFormMul(self.manifold)
        ret.forms_list = self.forms_list.copy()
        ret.factors = [simplify(f,kwargs=kwargs) for f in self.factors]
        
        ret.remove_squares()
        ret.remove_above_top()
        ret.sort_form_sums()
        ret.collect_forms()

        r = ret.__is_number()
        if r != None: return r

        return ret
    
    def subs(self, target, sub = None) -> DifferentialFormMul:
        """Substitute factors or 1 term differential forms in a generic differential form. """
        ret = DifferentialFormMul(self.manifold)
        ret.factors = deepcopy(self.factors)
        ret.forms_list = deepcopy(self.forms_list)

        if isinstance(target,DifferentialForm):
            new_forms_list = []
            new_factors_list = []
            for i in range(len(ret.forms_list)):
                if target in ret.forms_list[i]:
                    j = ret.forms_list[i].index(target)
                    if isinstance(sub,(float,int,AtomicExpr,Expr,Number)):
                        new_forms_list +=[ret.forms_list[i][:j] + ret.forms_list[i][j+1:]]
                        new_factors_list.append(ret.factors[i]*sub/target.factors[0])
                    elif isinstance(sub,DifferentialForm):
                        new_forms_list += [ret.forms_list[i][:j] + [sub] + ret.forms_list[i][j+1:]]
                        new_factors_list.append(ret.factors[i])
                    elif isinstance(sub,DifferentialFormMul):
                        for k in range(len(sub.factors)):
                            s = sub.forms_list[k]
                            f = sub.factors[k]
                            new_forms_list+= [ret.forms_list[i][:j] + s + ret.forms_list[i][j+1:]]
                            new_factors_list.append(ret.factors[i]*f)
                    else:
                        new_forms_list+=[ret.forms_list[i]]
                        new_factors_list.append(ret.factors[i])
                else:
                    new_forms_list+=[ret.forms_list[i]]
                    new_factors_list.append(ret.factors[i])
            ret.factors = new_factors_list
            ret.forms_list = new_forms_list
        elif isinstance(target,DifferentialFormMul):
            if len(target.factors) > 1: raise NotImplementedError("Cannot replace more than 1 term at a time")
            new_forms_list = []
            new_factors_list = []
            for i in range(len(ret.forms_list)):
                match_index = -1
                for j in range(len(ret.forms_list[i])-len(target.forms_list[0])+1):
                    if ret.forms_list[i][j:j+len(target.forms_list[0])] == target.forms_list[0]:
                        match_index = j
                        break
                if match_index != -1:
                    if isinstance(sub,DifferentialFormMul):
                        for k in range(len(sub.factors)):
                            s = sub.forms_list[k]
                            f = sub.factors[k]
                            new_forms_list += [ret.forms_list[i][:match_index] + s + ret.forms_list[i][match_index+len(target.forms_list[0]):]]
                            new_factors_list.append(ret.factors[i]*f/target.factors[0])
                    elif isinstance(sub,DifferentialForm):
                        new_forms_list += [ret.forms_list[i][:match_index] + [sub] + ret.forms_list[i][match_index+len(target.forms_list[0]):]]
                        new_factors_list.append(ret.factors[i]/target.factors[0])
                    elif isinstance(sub,(float,int,AtomicExpr,Expr)):
                        new_forms_list +=[ret.forms_list[i][:match_index] + ret.forms_list[i][match_index+len(target.forms_list[0]):]]
                        new_factors_list.append(ret.factors[i]*sub/target.factors[0])
                else:
                    new_forms_list += [ret.forms_list[i]]
                    new_factors_list.append(ret.factors[i])
            ret.factors = new_factors_list
            ret.forms_list = new_forms_list
        elif isinstance(target,dict):
            for key in target:
                ret = ret.subs(key,target[key])
        elif sub != None:
            for i in range(len(self.factors)):
                ret.factors[i] = ret.factors[i].subs(target,sub)
        
        ret.remove_squares()
        ret.remove_above_top()
        ret.sort_form_sums()
        ret.collect_forms()

        r = ret.__is_number()
        if r != None: return r

        return ret

    def to_tensor(self) -> Tensor:
        """Converts a DifferentialForm to a Tensor object. """
        ret = Tensor(self.manifold)
        for i in range(len(self.factors)):
            L = len(self.forms_list[i])
            for perm in permutations(list(range(L)),L):
                parity = int(Permutation(perm).is_odd)
                ret.comps_list += [[self.forms_list[i][p] for p in perm]]
                ret.factors += [(-1)**(parity)*self.factors[i]/factorial(L)]
        return factorial(self.get_degree())*ret

    def get_degree(self) -> int:
        """Returns the degree of a differential form. """
        weights = [sum(map(lambda x: x.get_degree(),f)) for f in self.forms_list]
        if len(set(weights)) == 1:
            return weights[0]
        return None

    def get_component_at_basis(self,basis=None):
        """Returns the compnent as a given basis of 1-forms. """
        basis_comp = basis
        if isinstance(basis,DifferentialFormMul):
            assert(len(basis.factors) == 1)
            assert(self.get_degree() == basis.get_degree())
            basis_comp = basis.forms_list[0]
        elif isinstance(basis,DifferentialForm):
            assert(self.get_degree() == 1)
            basis_comp = basis
        
        for i in range(len(self.forms_list)):
            f = self.forms_list[i]
            if f == basis_comp:
                return self.factors[i]
        return Number(0)

    def simplify(self, **kwargs):
        """ Returns the simplification of a differential form. """
        return self._eval_simplify(**kwargs)

    def apply_func_to_factors(self, func, **kwargs) -> DifferentialFormMul:
        """ Evaluate a sympy function on the factors of a differential form. """
        ret = DifferentialFormMul(self.manifold)
        ret.forms_list = self.forms_list.copy()
        ret.factors = [func(f,kwargs) for f in self.factors]
        
        ret.collect_forms()

        r = ret.__is_number()
        if r != None: return r

        return ret

    def factor(self, **kwargs): return self.apply_func_to_factors(factor, **kwargs)
    def expand(self, **kwargs): return self.apply_func_to_factors(expand, **kwargs)

    def conjugate(self):
        """Return the complex conjugate of a differential form. """
        ret = DifferentialFormMul(self.manifold)
        ret.forms_list = [[f.conjugate() for f in f_list] for f_list in self.forms_list]
        ret.factors = [conjugate(f) for f in self.factors]

        r = ret.__is_number()
        if r != None: return r
        return ret

def remove_latex_arguments(object : Expr) -> str:
    """ Remove the arguments from sympy functions and return the LaTeX string. """
    if hasattr(object,'atoms'):
        functions = object.atoms(Function)
        reps = {}

        for fun in functions:
            if hasattr(fun, 'name'):
                reps[fun] = Symbol(fun.name)
        object = object.subs(reps)
    latex_str = latex(object)
    return latex_str

def display_no_arg(object : VectorField | DifferentialForm | DifferentialFormMul | Tensor) -> None:
    """Display an object without the arguments in sympy functions. """
    latex_str = remove_latex_arguments(object)
    display(Math(latex_str))

def scalars(names : str, **args) -> Symbol:
    """Overrides the symbols class for sympy (probably not needed but useful for semantics). """
    return symbols(names,**args)

def differentialforms(manifold : Manifold, symbs : list, degrees : list) -> list[DifferentialForm]:
    """Returns a differential form given a list of symbols and degrees. """
    # TODO: Explain how this works with the different cases of symbols and degrees.
    ret = None
    if isinstance(symbs,str):
        ret = differentialforms(manifold,list(symbols(symbs)),degrees)
    elif isinstance(symbs,list):
        if isinstance(degrees,list):
            assert(len(symbs) == len(degrees))
            ret = [DifferentialForm(manifold,symbs[i],degrees[i]) for i in range(len(degrees))]
        elif isinstance(degrees,int):
            ret = [DifferentialForm(manifold,s,degrees) for s in symbs]
    elif isinstance(symbs,Symbol):
        if isinstance(degrees,list):
            ret = [DifferentialForm(manifold,symbs,d) for d in degrees]
        else:
            ret = DifferentialForm(manifold,symbs,degrees)
    else:
        raise NotImplementedError
    if isinstance(ret,list) and len(ret) == 1:
        return ret[0]
    return ret

def vectorfields(manifold : Manifold, symbs : list) -> list[VectorField]:
    """Returns vector fields corresponding to the symbols given, on the manifold provided. """
    ret = None
    if isinstance(symbs,str):
        ret = vectorfields(manifold,list(symbols(symbs)))
    elif isinstance(symbs,(list,tuple)):
        ret = [VectorField(manifold,s) for s in symbs]
    elif isinstance(symbs,Symbol):
        ret = [VectorField(manifold,symbs)]
    else:
        raise NotImplementedError
    if len(ret) == 1:
        return ret[0]
    return ret

def constants(names : str, **assumptions) -> Symbols:
    """ Uses the Quantity function to create constant symbols. """
    names = re.sub(r'[\s]+', ' ', names)
    consts = [Quantity(c,**assumptions) for c in names.split(' ') if c != '']
    if len(consts) == 1: return consts[0]
    return consts

def ExteriorDerivative(form : DifferentialFormMul | DifferentialForm | Expr, manifold : Manifold = None) -> DifferentialFormMul:
    """Computes the exterior derivative of differential forms. """
    if isinstance(form,(DifferentialForm,DifferentialFormMul)):
        return form.d
    
    elif isinstance(form,Expr):
        if manifold == None: raise NotImplementedError("Manifold cannot be None for Scalar input")
        ret = DifferentialFormMul(manifold)
        new_forms_list = []
        new_factors_list = []
        for f in form.free_symbols:
            dform = form.diff(f)
            if dform != 0:
                new_forms_list += [[DifferentialForm(manifold,f,0).d]]
                new_factors_list += [dform]
        
        ret.forms_list = new_forms_list
        ret.factors = new_factors_list
        if ret.forms_list == []:
            if len(ret.factors) == 1: return ret.factors[0]
            else:
                return Number(0)

        return ret

    elif isinstance(form,(float,int)):
        return Number(0)

    raise NotImplementedError

def ExteriorCoDerivative(form : DifferentialFormMul | DifferentialForm | Expr, manifold : Manifold = None) -> DifferentialFormMul:
    """Compute the codifferential operators on differential forms"""
    if isinstance(form,(DifferentialForm,DifferentialFormMul)):
        k = form.get_degree()
        n = form.manifold.dimension
        s = form.manifold.signature_prod
        return (-1)**(n*(k+1)+1)*s*Hodge(ExteriorDerivative(Hodge(form,manifold),manifold),manifold)
    n = manifold.dimension
    s = manifold.signature_prod
    return (-1)**(n+1)*s*Hodge(ExteriorDerivative(Hodge(form,manifold),manifold),manifold)

def PartialDerivative(tensor : Tensor, manifold : Manifold = None) -> Tensor:
    """Computes the partial derivative of an object on the manifold provided. """
    if isinstance(tensor,(DifferentialForm,DifferentialFormMul)):
        return PartialDerivative((1*tensor).to_tensor(),manifold)
    elif isinstance(tensor,(VectorField)):
        return Number(0)
    elif isinstance(tensor,(AtomicExpr,Expr,Function)):
        if manifold == None: raise NotImplementedError("Manifold cannot be None for Scalar input")
        ret = Tensor(manifold)
        for i in range(manifold.dimension):
            ret.comps_list += [[manifold.basis[i]]]
            ret.factors += [tensor.diff(manifold.coords[i])]
        ret._collect_comps()
        return ret
    elif isinstance(tensor,Tensor):
        ret = Tensor(tensor.manifold)
        man = tensor.manifold
        for i in range(man.dimension):
            for j in range(len(tensor.factors)):
                ret.comps_list += [[man.basis[i]]+tensor.comps_list[j]]
                ret.factors += [diff(tensor.factors[j],man.coords[i])]
        ret._collect_comps()
        return ret
    return Number(0)

def CovariantDerivative(tensor : Tensor,manifold : Manifold = None) -> Tensor:
    """Computes the covariant derivative, with respect to the metric, on the manifold provided. """
    if isinstance(tensor,(DifferentialForm,DifferentialFormMul)):
        return CovariantDerivative((Number(1)*tensor).to_tensor(),manifold)
    elif isinstance(tensor,VectorField):
        return CovariantDerivative(Number(1)*tensor)
    elif isinstance(tensor,(AtomicExpr,Expr,Function)):
        if manifold == None: raise NotImplementedError("Manifold cannot be None for Scalar input")
        ret = Tensor(manifold)
        for i in range(manifold.dimension):
            ret.comps_list += [[manifold.basis[i]]]
            ret.factors += [tensor.diff(manifold.coords[i])]
        ret._collect_comps()
        return ret
    elif isinstance(tensor,Tensor):
        t_weight = tensor.get_weight()
        Gamma = tensor.manifold.get_christoffel_symbols()
        Gamma_tensor = Gamma*tensor
        CD_tensor = PartialDerivative(tensor)
        for i in range(len(t_weight)):
            if t_weight[i] == -1:
                index_list = [0] + list(range(2,len(t_weight)+1))
                index_list.insert(i+1,1)
                CD_tensor += -PermuteIndices(Contract(Gamma_tensor,(0,3+i)),index_list)
            elif t_weight[i] == 1:
                index_list = list(range(1,len(t_weight)+1))
                index_list.insert(i+1,0)
                CD_tensor += PermuteIndices(Contract(Gamma_tensor,(2,3+i)),index_list)
        return CD_tensor

def WedgeProduct(left : DifferentialFormMul | DifferentialForm, right : DifferentialFormMul | DifferentialForm) -> DifferentialFormMul:
    """Wedge product multiplication for differential forms. """
    ret = None
    if isinstance(left,(int,float,Number,AtomicExpr,Expr)):
        left = left if not isinstance(left,(int,float)) else Number(left)
        if left == 0:
            return Number(0)
        if isinstance(right,(int,float,Number,AtomicExpr,Expr)):
            right = right if not isinstance(right,(int,float)) else Number(right)
            return left*right
        elif isinstance(right,DifferentialForm):
            ret = DifferentialFormMul(right.manifold)
            ret.forms_list = [[right]]
            ret.factors = [left]
        elif isinstance(right,DifferentialFormMul):
            ret = DifferentialFormMul(right.manifold)
            ret.forms_list = right.forms_list[:]
            ret.factors = [left*f for f in right.factors]
        else:
            raise NotImplementedError
    elif isinstance(left, DifferentialForm):
        ret = DifferentialFormMul(left.manifold)
        if isinstance(right,(int,float,Number,AtomicExpr,Expr)):
            if right == 0: return Number(0)
            ret.forms_list = [[left]]
            ret.factors = [right if not isinstance(right,(int,float)) else Number(right)]
        elif isinstance(right,DifferentialForm):
            assert(right.manifold == left.manifold)
            ret.forms_list = [[left,right]]
            ret.factors = [1]
        elif isinstance(right,DifferentialFormMul):
            assert(right.manifold == left.manifold)
            ret.forms_list = [[left]+rf for rf in right.forms_list]
            ret.factors = right.factors[:]
        else:
            raise NotImplementedError
    elif isinstance(left,DifferentialFormMul):
        ret = DifferentialFormMul(left.manifold)
        if isinstance(right,(int,float,Number,AtomicExpr,Expr)):
            if right == 0:
                return Number(0)
            ret.forms_list = left.forms_list
            right = right if not isinstance(right,(int,float)) else Number(right)
            ret.factors = [right*f for f in left.factors]
        elif isinstance(right,DifferentialForm):
            assert(left.manifold == right.manifold)
            ret.forms_list = [lf+[right] for lf in left.forms_list]
            ret.factors = left.factors[:]
        elif isinstance(right,DifferentialFormMul):
            assert(left.manifold == right.manifold)
            for i in range(len(left.forms_list)):
                for j in range(len(right.forms_list)):
                    ret.forms_list.append(left.forms_list[i]+right.forms_list[j])
                    ret.factors.append(left.factors[i]*right.factors[j])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    ret.remove_squares()
    ret.remove_above_top()
    ret.sort_form_sums()
    ret.collect_forms()

    if ret.factors == [] and ret.forms_list == []: 
        ret.factors = [Number(0)]
        ret.forms_list = [[]]
    return ret

def MetricWedgeProduct(left : DifferentialFormMul | DifferentialForm, right : DifferentialFormMul | DifferentialForm) -> DifferentialFormMul:
    """ Computes the Wedge product after a single contraction using the metric """
    man         = left.manifold
    frame_vects = man.get_inverse_frame()   
    sign        = man.signature

    return sum([sign[I]*left(frame_vects[I])*right(frame_vects[I]) for I in range(man.dimension)])

def TensorProduct(left : Tensor, right : Tensor) -> Tensor:
    """Tensor product of two objects on the same manifold. """
    if isinstance(left,DifferentialFormMul) or isinstance(right,DifferentialFormMul): raise NotImplementedError("Must convert DifferentialFormMul into Tensor before using with TensorProduct")
    ret = None
    if isinstance(left,(int,float,AtomicExpr,Expr)):
        left = left if not isinstance(left,(int,float)) else Number(left)
        if left == 0: return Number(0)
        if isinstance(right,(int,float,AtomicExpr,Expr)):
            right = right if not isinstance(right,(int,float)) else Number(right)
            return left*right
        elif isinstance(right,(DifferentialForm,VectorField)):
            ret = Tensor(right.manifold)
            ret.comps_list = [[right]]
            ret.factors = [left]
        elif isinstance(right,Tensor):
            ret = Tensor(right.manifold)
            ret.comps_list = right.comps_list.copy()
            ret.factors = [left*f for f in right.factors]
        else:
            raise NotImplementedError
    elif isinstance(left,VectorField):
        ret = Tensor(left.manifold)
        if isinstance(right,(int,float,AtomicExpr,Expr)):
            if right == 0: return Number(0)
            ret.comps_list = [[left]]
            ret.factors = [right if not isinstance(right,(int,float)) else Number(right)]
        elif isinstance(right,(DifferentialForm,VectorField)):
            assert(left.manifold == right.manifold)
            ret.comps_list = [[left,right]]
            ret.factors = [1]
        elif isinstance(right,Tensor):
            assert(left.manifold == right.manifold)
            ret.comps_list = [[left]+f for f in right.comps_list]
            ret.factors = right.factors
        else:
            raise NotImplementedError
    elif isinstance(left,DifferentialForm):
        ret = Tensor(left.manifold)
        if isinstance(right,(int,float,AtomicExpr,Expr)):
            if right == 0: return Number(0)
            ret.comps_list = [[left]]
            ret.factors = [right if not isinstance(right,(int,float)) else Number(right)]
        elif isinstance(right,(DifferentialForm,VectorField)):
            assert(left.manifold == right.manifold)
            ret.comps_list = [[left,right]]
            ret.factors = [1]
        elif isinstance(right,Tensor):
            assert(left.manifold == right.manifold)
            ret.comps_list = [[left]+f for f in right.comps_list]
            ret.factors = right.factors
        else:
            raise NotImplementedError
    elif isinstance(left,Tensor):
        ret = Tensor(left.manifold)
        if isinstance(right,(int,float,AtomicExpr,Expr)):
            if right == 0: return Number(0)
            ret.comps_list = left.comps_list.copy()
            right = Number(right) if isinstance(right,(int,float)) else right
            ret.factors = [right*f for f in left.factors]
        elif isinstance(right,(DifferentialForm,VectorField)):
            assert(left.manifold == right.manifold)
            ret.comps_list = [f+[right] for f in left.comps_list]
            ret.factors = left.factors
        elif isinstance(right,Tensor):
            assert(left.manifold == right.manifold)
            ret.comps_list = []
            for i in range(len(left.comps_list)):
                for j in range(len(right.comps_list)):
                    ret.comps_list += [left.comps_list[i]+right.comps_list[j]]
                    ret.factors += [left.factors[i]*right.factors[j]]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    ret._collect_comps()

    if ret.comps_list == [[]] or ret.comps_list == []:
        return Number(0) if ret.factors == [] else ret.factors[0]
    return ret

def Contract(tensor : Tensor,*positions : list[int]) -> Tensor:
    """Contract two tensors, given a list of pairs of indices to contract. Contraction must be between a differential form and vector field. """
    if isinstance(tensor,(int,float,Expr)): return tensor
    elif not isinstance(tensor,Tensor): raise TypeError("First argument must be a Tensor.")
    if tensor.comps_list == [[]] or tensor.comps_list == []:
        return Number(0) if tensor.factors == [] else  tensor.factors[0]
    tensor_weight = tensor.get_weight()
    if tensor_weight == (None): raise TypeError("Tensors must be of consistent types")
    p1_list = []
    p2_list = []
    for p in positions:
        p1,p2 = p
        p1_list += [p1]
        p2_list += [p2]
        if p1 > len(tensor_weight) or p2 > len(tensor_weight) or p1 < 0 or p2 < 0: raise IndexError("Contraction index out of range.")
        if tensor_weight[p1]*tensor_weight[p2] == 1: raise NotImplementedError("Tensor Contraction must be between vector fields and differential forms components.")
    ret = Tensor(tensor.manifold)
    max_index = len(tensor.factors)
    for i in range(max_index):
        left_popped = []
        right_popped = []
        total_without = []
        for k,e in enumerate(tensor.comps_list[i]):
            if k in p1_list: left_popped.append(e)
            if k in p2_list: right_popped.append(e)
            if not k in p1_list and not k in p2_list:
                total_without.append(e)
        
        sign = 1
        for k in range(len(left_popped)):
            if isinstance(left_popped[k],DifferentialForm):
                sign *= left_popped[k].insert(right_popped[k])
            else:
                sign *= right_popped[k].insert(left_popped[k])
        if sign != 0:
            ret.comps_list += [total_without]
            ret.factors += [tensor.factors[i]*sign]
    ret._collect_comps()
    if ret.comps_list ==[[]]: return ret.factors[0]
    if ret.comps_list == []: return Number(0)
    return ret

def PermuteIndices(tensor : Tensor, new_order : list[int]) -> Tensor:
    """Permute the basis elements of a tensor, given a new basis order. """
    if isinstance(tensor,(int,float,Number)): return tensor
    t_weight = tensor.get_weight()
    if (len(new_order)!=len(t_weight)): raise NotImplementedError("New index order must contain every index")
    if set(new_order) != set(range(len(t_weight))): raise TypeError("New index order does not contain every index once and only once")
    ret = Tensor(tensor.manifold)
    for i in range(len(tensor.factors)):
        ret.factors += [tensor.factors[i]]
        ret.comps_list += [[tensor.comps_list[i][j] for j in new_order]]
    
    ret._collect_comps()
    return ret

def LieDerivative(vector : Tensor | VectorField, tensor : Tensor | DifferentialFormMul | DifferentialForm, weight : int = 0) -> Tensor | DifferentialFormMul:
    """Compute the Lie derivative of a tensor given a vector field. """
    if not isinstance(vector,(Tensor,VectorField)): raise TypeError("First argument for the Lie derivative must be a vector")
    if isinstance(vector,Tensor) and not vector.is_vectorfield(): return TypeError("First argument for the Lie derivative must be a vector")

    if isinstance(tensor,(DifferentialFormMul,DifferentialForm)):
        ExtDTensor = ExteriorDerivative(tensor)
        DivVector = Contract(PartialDerivative(vector),(0,1))
        return ExteriorDerivative(tensor.insert(vector),tensor.manifold) + (Number(0) if ExtDTensor == 0 else ExtDTensor.insert(vector)) + weight*DivVector
    elif isinstance(tensor,VectorField):
        DivVector = Contract(PartialDerivative(vector),(0,1))
        return -sum([tensor(vector.factors[i])*vector.comps_list[i][0] for i in range(len(vector.factors))]) + weight*DivVector
    elif isinstance(tensor,Tensor):
        LieD_tensor = Contract(vector*PartialDerivative(tensor),(0,1))
        PDvector = PartialDerivative(vector)
        DivVector = Contract(PDvector,(0,1))
        if PDvector == 0: return LieD_tensor
        tensor_weights = tensor.get_weights_list()
        for I in range(len(tensor.factors)):
            term = tensor.get_sub_tensor(I)
            for i in range(len(tensor.comps_list[I])):
                sign = -tensor_weights[I][i]
                new_indices = list(range(len(tensor.comps_list[I])))
                j = len(tensor.comps_list[I]) + (1 if sign == 1 else 0)
                new_indices[i] = j-2
                new_indices[j-2] = i
                LieD_tensor += sign*PermuteIndices(Contract(term*PDvector,(i,j)),new_indices)
        return LieD_tensor + weight*DivVector*tensor
    raise NotImplementedError("Only the Tensor class and the Differential form class can be acted on by the LieDerivative")

def FormsListInBasisMatrix(formslist : dict, basis : DifferentialForm = None) -> Matrix:
    """Create a matrix from a list of 1-form and basis forms. The components of the matrix are the factors for each basis element in the 1-forms in a block matrix. """
    if basis == None:
        if formslist[0].manifold.basis == None: raise NotImplementedError("Need to set a basis for the manifold.")
        basis = formslist[0].manifold.basis
    
    from itertools import chain
    basis_comp_all = list(chain(*[list(chain(*(b.forms_list))) for b in basis]))

    basis_comp = []
    for bc in basis_comp_all:
        if bc not in basis_comp: basis_comp.append(1*bc)

    basis_comp_matrix = Matrix([[b.get_component_at_basis(bc) for bc in basis_comp] for b in basis])

    basis_comp_matrix_inv = basis_comp_matrix.inv()
    
    form_matrix = Matrix([[f.get_component_at_basis(b) for b in basis_comp] for f in formslist])

    return_matrix = form_matrix*basis_comp_matrix_inv

    return return_matrix

def Rank2TensorInverse(tensor : Tensor) -> Tensor:
    """ Computes the inverse of a rank-2 tensor with any index structure """
    weight = tensor.get_weight()
    assert(len(weight) == 2)
    man = tensor.manifold
    basis_vects = [man.get_basis(), man.get_vectors()]
    left  = basis_vects[-weight[0]]
    right = basis_vects[-weight[1]]
    component_array = [[0 for  _ in range(man.dimension)] for _ in range(man.dimension)]
    for I in range(man.dimension):
        for J in range(I,man.dimension):
            component_array[I][J] = Contract(tensor*left[I]*right[J],(0,2),(1,3))
    components_matrix = Matrix(component_array)
    matrix_inv = components_matrix.inv()
    ret = sum([matrix_inv[I,J]*left[I]*right[J] for I,J in drange(man.dimension,2)])
    return ret

def Hodge(form : DifferentialFormMul, M : Manifold = None,orientation : int = 1) -> DifferentialFormMul:
    """Computes the hodge star of a differential form given the corresponding manifold has a metric and basis 1-forms defined. """
    if isinstance(form,(int,float,Expr)):
        if M == None: raise(TypeError("Manifold must be specified for Hodge Dual of a Scalar"))
        return -orientation*M.signature_prod*form*M.get_volume_form()
    
    if form.manifold.coords == None:
        raise(NotImplementedError("Coordinate free Hodge star operator not implemeneted yet"))
    degree     = form.get_degree()
    dim        = form.manifold.dimension
    new_degree = dim-degree
    signature  = form.manifold.signature_prod

    # Fast differential form calculation
    g_UU = form.manifold.get_inverse_metric()
    ret  = None
    for I in range(len(form.forms_list)):
        term           = form.forms_list[I]
        fact           = form.factors[I]
        insert_vectors = [Contract(g_UU*p.to_tensor(),(1,2)) for p in term]
        ret_term       = -orientation*signature*fact*form.manifold.get_volume_form()
        for v in insert_vectors:
            ret_term = ret_term.insert(v)
        if ret == None:
            ret = ret_term
        else:
            ret = ret + ret_term
    return ret

def GetChristoffelSymbols(metric : Tensor, vectors : list[VectorField]) -> Tensor:
    if isinstance(metric,Tensor) and metric.get_weight() == (-1,-1): pass
    else: raise NotImplementedError("Argument: 'metric' must by a tensor of weight (-1,-1).")
    if vectors == None:
        vectors = metric.manifold.get_vectors()
    metric_UU = Rank2TensorInverse(metric)
    T_DDD = PartialDerivative(metric)
    g_UU_T_DDD = metric_UU*T_DDD
    Gamma_UDD_1 = Contract(g_UU_T_DDD,(1,3))
    return simplify((Gamma_UDD_1 + PermuteIndices(Gamma_UDD_1,(0,2,1)) - Contract(g_UU_T_DDD,(1,2)))/Number(2)).simplify()

def GetRiemannCurvature(metric : Tensor = None, christoffel_symbols : Tensor = None, vectors : Tensor = None) -> Tensor:
    if metric != None:
        if christoffel_symbols == None:
            christoffel_symbols = GetChristoffelSymbols(vectors,metric)
    else:
        if christoffel_symbols == None:
            raise(NotImplementedError("Either metric or christoffel symbols must be supplied to compute Riemann curvature"))
    
    christoffel_symbols = self.get_christoffel_symbols()
    dG = PartialDerivative(christoffel_symbols)
    Riemann = PermuteIndices(dG,(1,3,0,2)) + PermuteIndices(Contract(christoffel_symbols*christoffel_symbols,(2,3)),(0,3,1,2))
    return (Riemann - PermuteIndices(Riemann,(0,1,3,2)))

def GetRicciCurvature(metric : Tensor = None, christoffel_symbols : Tensor = None, riemann_tensor : Tensor = None) -> Tensor:
    if riemann_tensor == None:
        riemann_tensor = GetRiemannCurvature(metric,christoffel_symbols)
    else:
        raise NotImplementedError("Metric, Christoffel Symbols or Riemann tensor are not provided.")
    return Contract(riemann_tensor,(0,2))

def GetRicciScalar(metric : Tensor = None, christoffel_symbols : Tensor = None, riemann_tensor : Tensor = None, ricci_curvature : Tensor = None) -> Tensor:
    if metric == None:
        raise NotImplementedError("Metric needs to be provided to compute trace of curvature")
    metric_inverse = Rank2TensorInverse(metric)
    if ricci_curvature == None:
        ricci_curvature = GetRicciCurvature(metric=metric,christoffel_symbols=christoffel_symbols,riemann_tensor=riemann_tensor)
    return Contract(ricci_curvature*metric_inverse,(0,2),(1,3))

def GetEinsteinTensor(metric : Tensor = None, christoffel_symbols : Tensor = None, riemann_tensor : Tensor = None, ricci_curvature : Tensor = None, ricci_scalar : Expr = None) -> Tensor:
    if metric == None:
        return NotImplementedError("Metric needed to compute Einstein tensor")
    if ricci_scalar == None:        
        ricci_scalar = GetRicciScalar(metric=metric,riemann_tensor=riemann_tensor,ricci_curvature=ricci_curvature)
    if ricci_curvature == None:
        ricci_curvature = GetRicciCurvature(metric=metric,christoffel_symbols=christoffel_symbols,riemann_tensor=riemann_tensor)
    
    return ricci_curvature - Number(1,2)*metric*ricci_scalar
    
def GetSpinConnection(frame : list[DifferentialFormMul]) -> list[list[DifferentialFormMul]]:
    man = frame.manifold
    sig = man.signature
    dim = man.dimension
    bas = man.get_basis()
    η = diag(signature)

    wIJ_K_symbols = Array([[[(+1 if I < J else -1)*symbols(fr"\omega^{{{min(I,J)}{max(I,J)}}}_{{{K}}}") if I!=J else 0 for K in range(dim)] for J in range(dim)] for I in range(dim)])
    wI_J_K_symbols = Array([[[sum([wIJ_K_symbols[I,L,K]*η[L,J] for L in range(dim)]) for K in range(dim)] for J in range(dim)] for I in range(dim)])
    spin_connection = [[sum([wI_J_K_symbols[I,J,K]*bas[K] for K in range(dim)]) for J in range(dim)] for I in range(dim)]
    torsion_equations = [ExteriorDerivative(frame[I]) + sum([spin_connection[I][J]*frame[J] for J in range(dim)]) for I in range(dim)]
    all_symbols = []
    for I in range(dim):
        for J in range(I+1,dim):
            for K in range(dim):
                all_symbols.append(wIJ_K_symbols[I,J,K])
    all_equations = []
    for eq in torsion_equations:
        for fact in eq.factors:
            if fact != 0:
                all_equations.append(fact)
    spin_comps_sol = solve(all_equations,all_symbols)
    return spin_connection.subs(spin_comps_sol)

def GetSpinCurvature(spin_connection : list[list[DifferentialFormMul]]) -> list[list[DifferentialFormMul]]:
    man = None
    for first_list in spin_connection:
        for form in first_list:
            if isinstnace(form,DifferentialFormMul): 
                man = form.manifold 
                break
    dim = man.manifold
    return [[ExteriorDerivative(spin_connection[I][J]) + sum([spin_connection[I][K]*spin_connection[K][J] for K in range(dim)]) for J in range(dim)] for I in range(dim)]