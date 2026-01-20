from sympy import Symbol, I, Integer, AtomicExpr, Rational, latex, Number, Expr, symbols, simplify, Function, LeviCivita, solve, Matrix, variations, factorial, Matrix, conjugate, Symbol, diff, factor, expand, simplify, Array, eye
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
#   Generalise manifold to any signature dimension
#   Find a better way to describe the VectorField class

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
        - tetrads_inv(List[DifferentialForm/DifferentialFormMul]): Inverse tetrads as a list of VectorFields.
        - metric(Tensor):                                          The metric on the manifold
        - metric_inv(Tensor):                                      Inverse of the metric
        - vectors(List[Tensor,VectorField]):                       List of Vectors/VectorFields that form a basis of the tangent space.
        - christoffel_symbols(Tensor):                             Christoffel symbols for the metric, def "get_christoffel_symbols" defines this.
    """
    def __init__(self,label:str,dimension:int,signature:int=1):
        """Initialise the Manifold
        
        Arguments:
            - label(String):      Arbitrary name for the manifold.
            - dimension(Integer): The dimension.
            - signature(Integer): Signature of the manifold.
        """
        self.label = label
        assert(dimension > 0)
        self.dimension = dimension
        self.signature = signature
        self.coords = None
        self.basis = None
        self.tetrads = None
        self.tetrads_inv = None
        self.metric = None
        self.metric_inv = None
        self.vectors = None
        self.christoffel_symbols = None
        self.riemann_curvature = None
        self.epsilon_tensor = None
        self.volume = None
        self.volume_form = None
    
    def __eq__(self,other):
        """Equates Manifolds by their label, dimension and signature."""
        if isinstance(other,Manifold):
            return (self.label == other.label) and (self.dimension == other.dimension) and (self.signature == other.signature)
        return False
    
    def __len__(self):
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
    
    def set_frame(self,tetrads,compute_metric=True) -> None:
        """Sets the tetrad variable to a list of 1-forms. Also creates the metric and inverse metric.
        
        Arguments:
            - tetrads(List[DifferentialForm/DifferentialFormMul]): List of 1-forms
        """
        self.tetrads = tetrads

        tetrads_D = [e.to_tensor() for e in tetrads]
        self.metric = self.signature*tetrads_D[0]*tetrads_D[0] + sum([tetrads_D[i]*tetrads_D[i] for i in range(1,self.dimension)])

        self.christoffel_symbols = None

        if self.vectors is not None:
            volume = prod(tetrads)
            for v in self.vectors:
                volume = volume.insert(v)
            self.volume = volume

        if compute_metric:
            tetrad_matrix = Matrix([[e.insert(v) for v in self.vectors] for e in tetrads])
            tetrad_matrix_inv = tetrad_matrix.inv().T
            self.tetrads_inv = [sum([tetrad_matrix_inv[I,u]*self.vectors[u] for u in range(self.dimension)]) for I in range(self.dimension)]
            self.metric_inv = self.signature*self.tetrads_inv[0]*self.tetrads_inv[0] + sum([self.tetrads_inv[i]*self.tetrads_inv[i] for i in range(1,self.dimension)])

    def get_frame(self):
        """Returns the list of frames"""
        return self.tetrads

    def get_inverse_frame(self):
        """Return the list of inverse frames"""
        return self.tetrads_inv
    
    def get_volume(self):
        """Return volume element"""
        return self.volume

    def get_volume_form(self):
        """ Computes and returns the volume form """
        if self.volume_form == None:
            self.volume_form = prod(self.tetrads)
        return self.volume_form

    def get_basis(self):
        """ Returns the Manifold 1-forms basis."""
        return self.basis

    def get_vectors(self):
        """Returns the Manifold VectorField basis."""
        return self.vectors

    def get_metric(self):
        """Returns the Manifold metric."""
        return self.metric
    
    def get_inverse_metric(self): 
        """Returns the inverse metric for the Manifold"""
        return self.metric_inv

    def get_christoffel_symbols(self,metric=None):
        """ Returns the Christoffel symbols for the metric, calculates the Christoffel symbols for the metric if need be.""" 
        if metric != None:
            if isinstance(metric,Tensor) and metric.get_weight() == (-1,-1): pass
            else: raise NotImplementedError("Arugment: 'metric' must by a tensor of weight (-1,-1).")
            metric_matrix_inv = Matrix([[Contract(metric*u*v,(0,2),(1,3)) for v in self.vectors] for u in self.vectors]).inv()
            metric_UU = sum([sum([metric_matrix_inv[i,j]*self.vectors[i]*self.vectors[j] for j in range(self.dimension)]) for i in range(self.dimension)])
            T_DDD = PartialDerivative(metric)
            g_UU_T_DDD = (metric_UU*T_DDD)
            Gamma_UDD_1 = Contract(g_UU_T_DDD,(1,3))
            return simplify((Gamma_UDD_1 + PermuteIndices(Gamma_UDD_1,(0,2,1)) - Contract(g_UU_T_DDD,(1,2)))/Number(2)).simplify()
        if self.christoffel_symbols == None:
            T_DDD = PartialDerivative(self.metric)
            g_UU_T_DDD = (self.metric_inv*T_DDD)
            Gamma_UDD_1 = Contract(g_UU_T_DDD,(1,3))
            self.christoffel_symbols = ((Gamma_UDD_1 + PermuteIndices(Gamma_UDD_1,(0,2,1)) - Contract(g_UU_T_DDD,(1,2)))/Number(2)).simplify()
        return self.christoffel_symbols

    def get_riemann_curvature_tensor(self):
        if self.riemann_curvature == None:
            G_UDD = self.get_christoffel_symbols()
            dG_DUDD = PartialDerivative(G_UDD)
            R_UDDD = PermuteIndices(dG_DUDD,(1,3,0,2)) + PermuteIndices(Contract(G_UDD*G_UDD,(2,3)),(0,3,1,2))
            self.riemann_curvature = (R_UDDD - PermuteIndices(R_UDDD,(0,1,3,2))).simplify()
        return self.riemann_curvature

    def get_levi_civita_symbol(self):
        """Return totally antisymmetric tensor"""
        if self.epsilon_tensor == None:
            self.epsilon_tensor = Tensor(self)
            for indices in permutations(list(range(self.dimension))):
                self.epsilon_tensor.comps_list.append([self.vectors[i] for i in indices])
                self.epsilon_tensor.factors.append(LeviCivita(*indices))
        return self.epsilon_tensor

    def get_selfdual_twoforms(self,orientation:int=1):
        """For a 4 dimensional Manifold, return the self-dual 2-forms in a given.
        
        Arguments:
            - orientation(Integer): Changes which self-dual half the 2-forms are calculated in (i.e. anti-self-dual or self-dual).
        """
        assert(len(self.tetrads)==4)
        sigma = 1 if self.signature == 1 else I
        self.selfdual_twoforms = [self.tetrads[0]*self.tetrads[i+1]*sigma-sum([int(LeviCivita(i,j,k))*self.tetrads[j+1]*self.tetrads[k+1] for j,k in drange(3,2)])*orientation/Number(2) for i in range(3)]
        return self.selfdual_twoforms
    
    def get_selfdual_connections(self,twoforms,dsubs=None,basis=None,orientation=1):
        """ Returns the self-dual connection corresponding to a triple of self-dual 2-forms. 
        
        Arguments:
            - twoforms(List[DifferentialForm]): List of the triple of two forms
        
        Returns:
            - List of self-dual connections as 3 DifferentialFormMul
        """
        star_dS_i = [Hodge(d(si)) for si in twoforms]
        J1_star_dS_i = J1(star_dS_i,twoforms)
        sigma = -Number(1) if self.signature == 1 else I
        return [orientation*sigma/Number(2)*(J1_star_dS_i[i] - orientation*star_dS_i[i]) for i in range(3)]

    def get_spin_connection(self,frame=None):
        """Computes the spin connection for a given frame in n-dimensions"""
        if frame == None:
            if self.tetrads == None: raise NotImplementedError("Basis unknown, set Manifold basis or pass basis as argument")
            frame = self.tetrads[:]
        wIJ_K_symbols = Array([[[symbols(fr"\omega^{{{I}{J}}}_{{{K}}}") for K in range(self.dimension)] for J in range(self.dimension)] for I in range(self.dimension)])
        id3 = eye(self.dimension)
        if self.signature == -1: id3[0,0] = -1
        wI_J_K_symbols = Array([[[[wIJ_K_symbols[I,L,K]*id3[L,J] for L in range(self.dimension)] for K in range(self.dimension)] for J in range(self.dimension)] for I in range(self.dimension)])
        spin_connection = [[sum([wI_J_K_symbols[I,J,K]*self.basis[K] for K in range(self.dimension)]) for J in range(self.dimension)] for I in range(self.dimension)]
        # The above line doesn't compile, why not?!
        # conds = [d(frame[I]) + sum([spin_connection[I,J]*self.frame[J] for J in range(self.dimension)]) for I in range(self.dimension)]

        # equations = []
        # for I in range(self.dimension):
        #     for comp in conds[I].factors:
        #         equations.append(comp)
        # eq_sol = solve(equations, wIJ_K_symbols)
        # for I,J in drange(spin_connection):
        #     spin_connection[I,J].subs(eq_sol)
        # return spin_connection
            
    def get_selfdual_curvatures(self,connections):


        """Returns the triple of self-dual curvatures, built from the self-dual connections. They are 2-forms.

        Arguments:
            - connections(List[DifferentialFormMul]): A triple of 1-forms 
        
        Returns:
            - List of 2-forms

        """
        return [d(connections[i],self) + sum([int(LeviCivita(i,j,k))*connections[j]*connections[k] for j,k in drange(3,2)])*Number(1,2) for i in range(3)]

    def get_selfdual_component_vector(self,twoform,selfdual):
        """Returns a 3 vector of self-dual components given a general 2-form

        Arguments:
            - twoform(List[DifferentialFormMul]):  Generic 2-form of which the self-dual matrix will be returned.
            - selfdual(List[DifferentialFormMul]): Triple of self-dual 2-forms.
        
        Returns:
            - Matrix of 3x3 components (Can be complex).
        """
        
        assert(twoform.get_degree() == 2)
        assert([s.get_degree() for s in selfdual] == [2, 2, 2])

        volSD = sum([s*s for s in selfdual]).get_factor(0)/(1 if selfdual[0].manifold.signature == 1 else I)
        return [(twoform*s).get_factor(0)/(2*volSD) for s in twoforms]

    def get_metric_determinant(self):
        """Returns the determinant of the metric.

        Returns:
            - Scalar (numpy expression) that is the determinant.
        """
        assert(self.vectors != None)
        assert(self.tetrads != None)
        detg = 1
        for e in self.tetrads:
            detg = detg*e
        
        return -detg.get_factor(0)**2

    def get_urbantke_metric(self,twoforms):
        S_i = S1,S2,S3 = twoforms
        S_iDD = [s.to_tensor() for s in S_i]
        assert(self.dimension == 4)

        sqrtdetg = simplify(sum([s*s for s in S_i]).get_factor(0))

        if sqrtdetg == 0: return 0

        Epsilon4D = sum([LeviCivita(u,v,r,s)*self.vectors[u]*self.vectors[v]*self.vectors[r]*self.vectors[s] for u,v,r,s in drange(4,4)])

        tS_iUU = [Contract(Epsilon4D*s_iDD,(2,4),(3,5))*Number(1,2) for s_iDD in S_iDD]

        f = 1 if self.signature == 1 else I
        g_DD = sum([Contract(S_iDD[i]*tS_iUU[k]*S_iDD[j],(1,2),(3,4))*LeviCivita(i,j,k) for i,j,k in drange(3,3)])
        return g_DD/sqrtdetg
        
class VectorField():
    """Class VectorField

    Symbolic single term vector representation.

    Attributes:
        - symbol(Symbol): The symbol with which the derivative of the vector field will be taken. 

    """
    def __init__(self,manifold:Manifold,symbol):
        """ Returns the vector field on a given Manifold and given the symbol that constitutes the derivative.

        Arguments:
            - manifold(Manifold): The manifold the vector will be associated too.
            - symbol(Symbol): The symbolic symbol that the derivative is taken with respect too.
        
        Returns:
            - VectorField
        """
        self.symbol = symbol
        self.manifold = manifold
    
    def __eq__(self,other):
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
        if isinstance(self,(int,float,AtomicExpr,Expr)):
            if other != 0:
                ret.comps_list.append([1])
                ret.factors.append(other)
        elif isinstance(other,(VectorField,DifferentialForm)):
            ret.comps_list.append([other])
            ret.factors(1)
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
        elif isinstance(other,float) or isinstance(other,int):
            if other != 0: ret = self + DifferentialForm(self.manifold,Rational(other),0)
        elif isinstance(other,AtomicExpr):
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
        return TensorProduct(self,1/Number(other))

    def __truediv__(self,other): 
        """True divide the tensor by a Scalar. """
        return TensorProduct(self,1/other)

    def __getitem__(self,indices):
        """Overloads the indexing operation to swap the order of the Tensor components.
        
        Arguments:
            - indices(List[Integer]): List of new order of indices that the Tensor bases should be.
        
        Return:
            Permuted Tensor.
         """
        s_weight = self.get_weight()
        if len(indices) != len(s_weight): raise IndexError("Indices not of correct form")
        if set(list(indices)) != set(range(len(s_weight))): raise IndexError("Each index must appear once and only once")
        if list(indices) == list(range(len(s_weight))): return self
        return PermuteIndices(self,list(indices))

    def _repr_latex_(self):
        """Returns the LaTeX String related to the Tensor. """
        if not _PRINT_ARGUMENTS:
            latex_str = "$" + "+".join([ "(" + remove_latex_arguments(self.factors[i]) + ")" + r" \otimes ".join([str(f) for f in self.comps_list[i]]) for i in range(len(self.comps_list))])  + "$"
        else: 
            latex_str = "$" + "+".join([ "(" + latex(self.factors[i]) + ")" + r" \otimes ".join([latex(f) for f in self.comps_list[i]]) for i in range(len(self.comps_list))])  + "$"
        if latex_str == "$$":
            return "$0$"
        return latex_str
    
    def is_vectorfield(self):
        """Check if a Tensor contains only VectorField components, and is valued in the Tangent space only. """
        for f in self.comps_list:
            if len(f) != 1 or not isinstance(f[0],VectorField):
                return False
        return True
    
    def get_weight(self):
        """ Returns the "weight" (weight is a number that ) """
        if len(self.factors) == 0: return (0,0)
        first_weight = tuple(map(lambda x: int(isinstance(x,VectorField))-int(isinstance(x,DifferentialForm)),self.comps_list[0]))
        for i in range(1,len(self.factors)):
            current_weight = tuple(map(lambda x: int(isinstance(x,VectorField))-int(isinstance(x,DifferentialForm)),self.comps_list[i]))
            if current_weight != first_weight: return (None)
        return first_weight

    def get_component(self,i:int):
        """Get the i-th component in the order that the factors list in in. """
        return self.factors[i]

    def _collect_comps(self):

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

    def subs(self,target,sub=None,simp=True):
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

    def get_factor(self,index:int):
        """Return the factor in the order of the factors list. 
        
        Arguments:
            - index(Integer): Index of the factor.

        Returns:
            Scalar that is the factor.
        """
        if len(self.factors) == 0: return 0
        return self.factors[index]
    
    def simplify(self):
        """ Simplify the factors of a component. """
        ret = Tensor(self.manifold)
        ret.factors = [simplify(f) for f in self.factors]
        ret.comps_list = self.comps_list.copy()
        ret._collect_comps()
        return ret

    def factor(self):
        """Factor the components of a factor. """
        ret = Tensor(self.manifold)
        ret.factors = [factor(f) for f in self.factors]
        ret.comps_list = self.comps_list.copy()
        ret._collect_comps()
        return ret
    
    def expand(self,deep=True, modulus=None,**hints):
        """Expand the components of a tensor. """
        ret = Tensor(self.manifold)
        ret.factors = [expand(f,deep=deep,modulus=modulus,**hints) for f in self.factors]
        ret.comps_list = self.comps_list.copy()
        ret._collect_comps()
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

    def conjugate(self):
        """ Return the complex conjugate of the Tensor. """
        ret = Tensor(self.manifold)
        ret.comps_list = [[f.conjugate() for  f in f_list] for f_list in self.comps_list]
        ret.factors = [conjugate(f) for f in self.factors]
        return ret

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

    def __init__(self,manifold,symbol,degree=0, exact=False):
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
        
    def __eq__(self,other):
        """ Compares if two differential forms are equal. """
        if not isinstance(other,DifferentialForm): return False
        return (self.symbol == other.symbol) and (self.get_degree() == other.get_degree())
    
    def __hash__(self): 
        """ Unique hash for a differential form. """
        return hash((str(self.symbol),self.get_degree()))

    def __mul__(self,other): 
        """ Multiplies the DifferentialForm with a Tensor/VectorField to produce a Tensor, or another DifferentialForm to produce a DifferentialFormMul. """
        if isinstance(other,(Tensor,VectorField)):
            return TensorProduct(self,other)
        return WedgeProduct(self,other)

    def __rmul__(self,other): 
        """Right multiplication of __mul__. """
        if isinstance(other,(Tensor,VectorField)):
            return TensorProduct(other,self)
        return WedgeProduct(other,self)
    
    def __div__(self,other):
        """Divide the DifferentialForm by another object. """
        return WedgeProduct(self,1/other)
    
    def __truediv__(self,other): 
        """Truediv version of __div__. """
        return WedgeProduct(self,1/other)

    def __add__(self,other):
        """Add together DifferentialForm and another object. """
        ret = DifferentialFormMul(self.manifold)
        ret.forms_list = [[self]]
        ret.factors = [1]
        if isinstance(other,(AtomicExpr,float,int,Number)):
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
    
    def __radd__(self,other): 
        """Right addition version of __add__. """
        return self + other

    def __lt__(self,other):
        """Less that operator to order differential forms. Ordered alphabetically by the String of the symbol. """
        if not isinstance(other,DifferentialForm): raise NotImplementedError
        if str(self.symbol) < str(other.symbol):
            return True
        elif str(self.symbol) > str(other.symbol):
            return False
        else:
            return (self.get_degree()) < other.get_degree()

    def __neg__(self):
        """Return the negative of a Differential Form. """
        return DifferentialFormMul(self.manifold,self,-1)
    
    def __sub__(self,other):
        """Subtract an object from the DifferentialForm. """
        return self + (-other)
    
    def __rsub__(self,other): 
        """Right subtraction version of __sub__. """
        return (-self) + other

    def __str__(self):
        """Returns the LaTeX String of the symbol. """
        return latex(self.symbol)

    def _repr_latex_(self):
        """Sympy internal call that returns the LaTeX string of the symbol. """
        return self.symbol._repr_latex_()

    def __hash__(self):
        """Unique hash for a DifferentialForm. """
        return hash((self.symbol,self.get_degree()))

    def to_tensor(self):
        """Converts a DifferentialForm to a Tensor, such that multiplication uses the TensorProduct instead of WedgeProduct. """
        return (Number(1)*self).to_tensor()
    
    __repr__ = _repr_latex_
    _latex   = _repr_latex_
    _print   = _repr_latex_
    
    def __eq__(self,other):
        """Tests if two DifferentialForms are equivalent. """
        if isinstance(other,DifferentialForm):
            return str(self.symbol) == str(other.symbol) and self.get_degree() == other.get_degree()
        return False

    def _eval_simplify(self, **kwargs):
        """Overrides sympy internal simplify call to return self. This object is already simplified by construction. """
        return self
    
    def insert(self,vector:VectorField):
        """ Insert a VectorField into the DifferentialForm. 
        
        Arguments:
            - vector(VectorField): The vector field that will be inserted into the DifferentialForm.
        
        Returns:
            Contraction of the DifferentialForm and VectorField as a Scalar.
        """
        if isinstance(vector,VectorField):
            if self.symbol == vector.symbol or str(self.symbol) == "d\\left("+str(vector.symbol)+"\\right)": return 1
            else: return Number(0)
        elif isinstance(vector,Tensor):
            if vector.is_vectorfield():
                return sum([vector.factors[i]*self.insert(vector.comps_list[i][0]) for i in range(len(vector.factors))])
        else:
            raise NotImplementedError

    @property
    def d(self):
        """ Exterior derivative of the differential form, in the given manifold. 
        
        Computes the Exterior Derivative of a differential form. If the differential field is purely symbol it returns zero if self.exact=True.
        Allows for purely symbolic differential forms by return a new differental form with symbol = "d(old_symbol)" which is exact (closed).

        Returns:
            DifferntialFormMul

        """

        if self.exact: return 0
        elif isinstance(self.symbol,Number): return 0
        else:
            dsymbol = symbols(r"d\left("+str(self.symbol)+r"\right)",**self.symbol.assumptions0)
            return DifferentialForm(self.manifold,dsymbol,degree=self.get_degree()+1,exact=True)
        raise NotImplementedError

    def subs(self,target,sub=None):
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

    def conjugate(self):
        """Return the complex conjugate of a DifferentialForm. """
        return DifferentialForm(self.manifold,conjugate(self.symbol),self.get_degree(),self.exact)

    def get_degree(self): 
        return self.degree

class DifferentialFormMul():
    """ Class: DifferentialFormMul

    Contains sums of products of the DifferentialForm class as a basis.

    Attributes:
        - forms_list(List[List[DifferentialForm]]): List of lists where the sub lists contaion the wedge product of DifferentialForms and the outer list represents the sum.
        - factors(List[Scalar]):                    List of factors for each term in the outer list of forms_list.
    """

    def __init__(self,manifold:Manifold,form:DifferentialForm=None,factor:AtomicExpr=None):
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
 
    def __add__(self,other):
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

        if ret.factors == [] and ret.forms_list == []: return 0
        elif ret.forms_list == [[]]: return ret.factors[0]
        return ret
    
    def __mul__(self,other): 
        """Multiply differential form with form/vectorfield or scalar. """
        if isinstance(other,(Tensor,VectorField)):
            return TensorProduct(self,other)
        return WedgeProduct(self,other)
    
    def __rmul__(self,other): 
        """Right multiplication version of __mul__. """
        if isinstance(other,(Tensor,VectorField)):
            return TensorProduct(other,self)
        return WedgeProduct(other,self)

    def is_Rational(self): return False
    def is_zero(self): return False

    def __div__(self,other): return WedgeProduct(self,(1/other))
    def __truediv__(self,other): return WedgeProduct(self,(1/other))

    def __radd__(self,other): return self + other
    def __neg__(self):
        ret = DifferentialFormMul(self.manifold)
        ret.forms_list = self.forms_list.copy()
        ret.factors = [-f for f in self.factors]
        return ret
    
    def __sub__(self,other): return self + (-other)
    def __rsub__(self,other): return other + (-self)

    def __eq__(self,other):
        """Checks if two differential forms are equivalent. """
        if isinstance(other,DifferentialForm) and self.factors == [1] and len(self.forms_list[0]) == 1: return other == self.forms_list[0][0]
        elif not isinstance(other,DifferentialFormMul): return False
        elif other.factors != self.factors: return False
        elif other.forms_list != self.forms_list: return False
        return True

    def __hash__(self): 
        """Unique hash for differential forms. """
        symbols = []
        for forms in self.forms_list: symbols+=forms
        symbols += self.factors
        return hash(tuple(symbols))

    def insert(self,other):
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

    def remove_squares(self):
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
        
    def remove_above_top(self):
        """Removes any differential form with degree above the top form. """
        i = 0
        while i < len(self.forms_list):
            if sum([f.get_degree() for f in self.forms_list[i]]) > self.manifold.dimension:
                del self.forms_list[i]
                del self.factors[i]
                continue
            i += 1

    def sort_form_sums(self):
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
    
    def collect_forms(self):
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
            
    def _repr_latex_(self):
        """Return the LaTeX String for a differential form. """
        if not _PRINT_ARGUMENTS:
            latex_str = "$" + "+".join([ "(" + remove_latex_arguments(self.factors[i]) + ")" + r" \wedge ".join([str(f) for f in self.forms_list[i]]) for i in range(len(self.forms_list))]) + "$"
        else:
            latex_str = "$" + "+".join([ "(" + latex(self.factors[i]) + ")" + r" \wedge ".join([str(f) for f in self.forms_list[i]]) for i in range(len(self.forms_list))]) + "$"
        if latex_str == "$$":
            return "$0$"
        return latex_str

    def get_factor(self,index):
        """Returns the factor at the index give, factors are in an arbitrary order. """
        if len(self.factors) == 0: return 0
        return self.factors[index]

    def get_degree(self):
        degree_set = set([sum([ssl.get_degree() for ssl in sl]) for sl in self.forms_list])
        if len(degree_set) == 1:
            return list(degree_set)[0]
        return None

    def __getitem__(self,index):
        #TODO: Make this permute the differential forms not index the components
        pass
    
    def __is_number(self):
        if self.forms_list == []:
            if len(self.factors) == 1: return self.factors[0]
            else:
                return 0
        return None
    
    _sympystr = _repr_latex_
    __str__ = _repr_latex_

    @property
    def d(self):
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

    def _eval_simplify(self, **kwargs):
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
    
    def subs(self,target,sub=None):
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

    def to_tensor(self):
        """Converts a DifferentialForm to a Tensor object. """
        ret = Tensor(self.manifold)
        for i in range(len(self.factors)):
            L = len(self.forms_list[i])
            for perm in permutations(list(range(L)),L):
                parity = int(Permutation(perm).is_odd)
                ret.comps_list += [[self.forms_list[i][p] for p in perm]]
                ret.factors += [(-1)**(parity)*self.factors[i]/factorial(L)]
        return factorial(self.get_degree())*ret

    def get_degree(self):
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
        return 0

    def simplify(self):
        """ Returns the simplification of a differential form. """
        return self._eval_simplify()

    def eval_sympy_func(self,func, **kwargs):
        """ Evaluate a sympy function on the factors of a differential form. """
        ret = DifferentialFormMul(self.manifold)
        ret.forms_list = self.forms_list.copy()
        ret.factors = [func(f,kwargs) for f in self.factors]
        
        ret.collect_forms()

        r = ret.__is_number()
        if r != None: return r

        return ret

    def factor(self):
        """Factor the components of a differential form. """
        ret = DifferentialFormMul(self.manifold)
        ret.forms_list = self.forms_list.copy()
        ret.factors = [simplify(f) for f in self.factors]

        r = ret.__is_number()
        if r != None: return r

        return ret    

    def expand(self):
        """Expand the component factors of a differential form. """
        ret = DifferentialFormMul(self.manifold)
        ret.factors = [f.expand() for f in self.factors]
        ret.forms_list = self.forms_list

        r = ret.__is_number()
        if r != None: return r

        return ret

    def conjugate(self):
        """Return the complex conjugate of a differential form. """
        ret = DifferentialFormMul(self.manifold)
        ret.forms_list = [[f.conjugate() for f in f_list] for f_list in self.forms_list]
        ret.factors = [conjugate(f) for f in self.factors]

        r = ret.__is_number()
        if r != None: return r
        return ret

def remove_latex_arguments(object):
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

def display_no_arg(object):
    """Display an object without the arguments in sympy functions. """
    latex_str = remove_latex_arguments(object)
    display(Math(latex_str))

def scalars(names:str,**args)->Symbol:
    """Overrides the symbols class for sympy (probably not needed but useful for semantics). """
    return symbols(names,**args)

def differentialforms(manifold:Manifold,symbs:list,degrees:list):
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

def vectorfields(manifold:Manifold,symbs:list):
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

def constants(names:str, **assumptions)->symbols:
    """ Uses the Quantity function to create constant symbols. """
    names = re.sub(r'[\s]+', ' ', names)
    consts = [Quantity(c,**assumptions) for c in names.split(' ') if c != '']
    if len(consts) == 1: return consts[0]
    return consts

def d(form,manifold=None):
    """Computes the exterior derivative of differential forms. """
    if isinstance(form,(DifferentialForm,DifferentialFormMul)):
        return form.d
    
    elif isinstance(form,(AtomicExpr,Expr,Function)):
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
                return 0

        return ret

    elif isinstance(form,(float,int)):
        return 0

    raise NotImplementedError

def cod(form,manifold=None):
    """Compute the codifferential operators on differential forms"""
    if isinstance(form,(DifferentialForm,DifferentialFormMul)):
        k = form.get_degree()
        n = form.manifold.dimension
        s = form.manifold.signature
        return (-1)**(n*(k+1)+1)*s*Hodge(d(Hodge(form,manifold),manifold),manifold)
    n = manifold.dimension
    s = manifold.signature
    return (-1)**(n+1)*s*Hodge(d(Hodge(form,manifold),manifold),manifold)

def PartialDerivative(tensor,manifold=None):
    """Computes the partial derivative of an object on the manifold provided. """
    if isinstance(tensor,(DifferentialForm,DifferentialFormMul)):
        return PartialDerivative((1*tensor).to_tensor(),manifold)
    elif isinstance(tensor,(VectorField)):
        return 0
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
    return 0

def CovariantDerivative(tensor,manifold=None):
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

def WedgeProduct(left,right,debug=False):
    """Wedge product multiplication for differential forms. """
    ret = None
    if isinstance(left,(int,float,Number,AtomicExpr,Expr)):
        left = left if not isinstance(left,(int,float)) else Number(left)
        if left == 0:
            return 0
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
            if right == 0: return 0
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
                return 0
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

def TensorProduct(left,right):
    """Tensor product of two objects on the same manifold. """
    if isinstance(left,DifferentialFormMul) or isinstance(right,DifferentialFormMul): raise NotImplementedError("Must convert DifferentialFormMul into Tensor before using with TensorProduct")
    ret = None
    if isinstance(left,(int,float,AtomicExpr,Expr)):
        left = left if not isinstance(left,(int,float)) else Number(left)
        if left == 0: return 0
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
            if right == 0: return 0
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
            if right == 0: return 0
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
            if right == 0: return 0
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
        return 0 if ret.factors == [] else ret.factors[0]
    return ret

def Contract(tensor,*positions):
    """Contract two tensors, given a list of pairs of indices to contract. Contraction must be between a differential form and vector field. """
    if isinstance(tensor,(int,float,Number)): return tensor
    elif not isinstance(tensor,Tensor): raise TypeError("First argument must be a Tensor.")
    if tensor.comps_list == [[]] or tensor.comps_list == []:
        return 0 if tensor.factors == [] else  tensor.factors[0]
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

def PermuteIndices(tensor,new_order):
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

def LieDerivative(vector,tensor):
    """Compute the Lie derivative of a tensor given a vector field. """
    if not isinstance(vector,(Tensor,VectorField)): raise TypeError("First argument for the Lie derivative must be a vector")
    if isinstance(vector,Tensor) and vector.get_weight() != (1,): return TypeError("First argument for the Lie derivative must be a vector")
    if isinstance(tensor,(DifferentialFormMul,DifferentialForm)):
        return d(tensor.insert(vector),tensor.manifold) + d(tensor).insert(vector)
    elif isinstance(tensor,VectorField):
        return 0
    elif isinstance(tensor,Tensor):
        LieD_tensor = Contract(vector*PartialDerivative(tensor),(0,1))
        PDvector = PartialDerivative(vector)
        if PDvector == 0: return LieD_tensor
        for I in range(len(tensor.factors)):
            for i in range(len(tensor.comps_list[I])):
                sign = 1 if isinstance(tensor.comps_list[I][i],DifferentialForm) else -1
                new_indices = list(range(len(tensor.comps_list[I])))
                j = len(tensor.comps_list[I]) + (1 if sign == 1 else 2)
                new_indices[i] = j-2
                new_indices[j-2] = i
                LieD_tensor += sign*PermuteIndices(Contract(tensor.get_component(I)*PDvector,(i,j)),new_indices)
        return LieD_tensor
    raise NotImplementedError("Only the Tensor class and the Differential form class can be acted on by the LieDerivative")

def FormsListInBasisMatrix(formslist:dict, basis=None) -> Matrix:
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
    
def Hodge(form : DifferentialFormMul,M=None,orientation=1) -> DifferentialFormMul:
    """Computes the hodge star of a differntial form given the corresponding manifold has a metric and basis 1-forms defined. """
    if not isinstance(form,(DifferentialForm,DifferentialFormMul)):
        if M == None: raise(TypeError("Manfold must be specified for Hodge Dual of a Scalar"))
        signature = M.signature
        return -orientation*signature*form*M.get_volume_form()
    
    if form.manifold.coords == None:
        raise(NotImplementedError("Coordinate free Hodge star operator not implemeneted yet"))
    degree = form.get_degree()
    dim = form.manifold.dimension
    new_degree = dim-degree
    signature = form.manifold.signature

    # Fast differential form calculation
    g_UU = form.manifold.get_inverse_metric()
    ret = None
    for I in range(len(form.forms_list)):
        term = form.forms_list[I]
        fact = form.factors[I]
        insert_vectors = [Contract(g_UU*p.to_tensor(),(1,2)) for p in term]
        ret_term = -orientation*signature*fact*form.manifold.get_volume_form()
        for v in insert_vectors:
            ret_term = ret_term.insert(v)
        if ret == None:
            ret = ret_term
        else:
            ret = ret + ret_term
    return ret

def J1(thetas,twoforms):
    """Computes the J1 operator on R3 valued 1-forms in 4D given a triple of 2-forms"""
    g_UU = twoforms[0].manifold.get_inverse_metric()
    S_iDU = [Contract(s.to_tensor()*g_UU,(1,2)).simplify() for s in twoforms]
    return [sum([LeviCivita(i,j,k)*Contract(S_iDU[j]*thetas[k].to_tensor(),(1,2)) for j,k in drange(3,2)]).to_differentialform() for i in range(3)]

def J2(Bi,twoforms):
    """Special product that exists in the context of SU(2) structures. """
    ### TODO: Simplify this and compact the notation
    Bi_DD = [2*b.to_tensor() for b in Bi]
    g_UU = twoforms[0].manifold.get_inverse_metric()
    Si_DU = [Contract(s.to_tensor()*g_UU,(1,2)) for s in twoforms]
    R1_DD = Contract(S2_DU*B3_DD - S3_DU*B2_DD,(1,2))
    R2_DD = Contract(S3_DU*B1_DD - S1_DU*B3_DD,(1,2))
    R3_DD = Contract(S1_DU*B2_DD - S2_DU*B1_DD,(1,2))

    return [sum([LeviCivita(i,j,k)*Contract(Si_DU*Bi_DD,(1,2)) for j,k in drange(3,2)]).to_differentialform()/Number(2) for i in range(3)]

def dA(thetas,A_i,manifold=None):
    result = [d(thetas[i],manifold) + sum([LeviCivita(i,j,k)*A_i[j]*thetas[k] for j,k in drange(3,2)]) for i in range(3)]
    return result

def codA(thetas,A_i,manifold=None):
    hodge_theta = [Hodge(t,manifold) for t in thetas]
    dA_hodge_theta = dA(hodge_theta,A_i,manifold)
    return [Hodge(t,manifold) for t in dA_hodge_theta]

def GetSelfDualTwoForm(frame,orientation=1,signature=1):
    assert(len(frame)==4)

    sigma = 1 if signature == 1 else I
    return [frame[0]*frame[i+1]*sigma-sum([int(LeviCivita(i,j,k))*frame[j+1]*frame[k+1] for j,k in drange(3,2)])*orientation/Number(2) for i in range(3)]

def GetSelfDualConnections(twoforms,signature=1):
    star_dS_i = [Hodge(d(si)) for si in twoforms]
    J1_star_dS_i = J1(star_dS_i,twoforms)
    sigma = Number(1) if signature == 1 else I
    return [orientation*sigma/Number(2)*(J1_star_dS_i[i] - orientation*star_dS_i[i]).simplify() for i in range(3)]