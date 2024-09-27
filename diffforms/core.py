from sympy import Symbol, I, Integer, AtomicExpr, Rational, latex, Number, Expr, symbols, simplify, Function, LeviCivita, solve, Matrix, variations, factorial, Matrix
from sympy.physics.units.quantities import Quantity
from IPython.display import Math
from sympy.combinatorics import Permutation
from itertools import permutations
import re
import numbers
from math import factorial, prod

class Manifold():
    def __init__(self,label,dimension,signature=1):
        self.label = label
        assert(dimension > 0)
        self.dimension = dimension
        self.signature = signature
        self.coords = None
        self.basis = None
        self.tetrads = None
        self.metric = None
        self.metric_inv = None
        self.vectors = None
        self.christoffel_symbols = None
    
    def __eq__(self,other):
        if isinstance(other,Manifold):
            return (self.label == other.label) and (self.dimension == other.dimension)
        return False
    
    def __len__(self): return self.dimension

    def set_coordinates(self,coordinates,signature=1):
        assert(len(coordinates) == self.dimension)
        self.coords = coordinates
        self.basis = [DifferentialForm(self,c,0).d for c in coordinates]
        self.signature = signature
        self.vectors = DefVectorFields(self,coordinates)
    
    def set_tetrads(self,tetrads):
        self.tetrads = tetrads
        tetrads_D = [e.to_tensor() for e in tetrads]
        self.metric = tetrads_D[0]*tetrads_D[0]
        for i in range(1,len(tetrads)):
            self.metric += tetrads_D[i]*tetrads_D[i]
        tetrad_matrix = Matrix([[e.insert(v) for v in self.vectors] for e in tetrads])
        tetrad_matrix_inv = tetrad_matrix.inv().T

        self.tetrads_inv = [sum([tetrad_matrix_inv[I,u]*self.vectors[u] for u in range(self.dimension)]) for I in range(self.dimension)]
        self.metric_inv = self.tetrads_inv[0]*self.tetrads_inv[0]
        for i in range(1,self.dimension):
            self.metric_inv += self.tetrads_inv[i]*self.tetrads_inv[i]
        
        T_DDD = PartialDerivative(self.metric)
        g_UU_T_DDD = (self.metric_inv*T_DDD)

        Gamma_UDD_1 = Contract(g_UU_T_DDD,(1,3))
        Gamma_UDD_2 = Gamma_UDD_1[0,2,1]
        Gamma_UDD_3 = Contract(g_UU_T_DDD,(1,2))

        self.christoffel_symbols = ((Gamma_UDD_1 + Gamma_UDD_2 - Gamma_UDD_3)/2).simplify()

    def get_basis(self): return self.basis
    def get_vectors(self): return self.vectors
    def get_metric(self): return self.metric
    def get_inverse_metric(self): return self.metric_inv
    def get_christoffel_symbols(self): return self.christoffel_symbols

    def get_selfdual_twoforms(self,orientation=1):
        assert(len(self.tetrads)==4)
        sigma = 1 if self.signature == 1 else I
        e0,e1,e2,e3 = self.tetrads
        S1 = simplify(sigma*e0*e1-orientation*e2*e3)
        S2 = simplify(sigma*e0*e2-orientation*e3*e1)
        S3 = simplify(sigma*e0*e3-orientation*e1*e2)
        self.selfdual_twoforms = [S1,S2,S3]
        return S1,S2,S3
    
    def get_selfdual_connections(self,twoforms):
            S1,S2,S3 = twoforms
            if self.basis == None: raise NotImplementedError("Basis unknown, set Manifold basis or pass basis as argument")

            A_symbols = symbols(r"A^{1:4}_{0:4}",real=True)

            A1 = sum([A_symbols[(1 -1)*4 + I]*self.basis[I] for I in range(4)])
            A2 = sum([A_symbols[(2 -1)*4 + I]*self.basis[I] for I in range(4)])
            A3 = sum([A_symbols[(3 -1)*4 + I]*self.basis[I] for I in range(4)])

            dAS1 = (d(S1) + A2*S3-A3*S2).simplify()
            dAS2 = (d(S2) + A3*S1-A1*S3).simplify()
            dAS3 = (d(S3) + A1*S2-A2*S1).simplify()

            A_solution = solve(dAS1.factors+dAS2.factors+dAS3.factors,A_symbols)

            A1 = A1.subs(A_solution).simplify()
            A2 = A2.subs(A_solution).simplify()
            A3 = A3.subs(A_solution).simplify()

            return A1,A2,A3

    def get_selfdual_curvatures(self,connections):
        A1,A2,A3 = connections
        F1 = d(A1) + A2*A3
        F2 = d(A2) + A3*A1
        F3 = d(A3) + A1*A2
        return F1,F2,F3

    def get_selfdual_curvature_matrix(self,curvatures,twoforms):
        S1,S2,S3 = twoforms
        sigma = 1 if S1.manifold.signature == 1 else I
        volS = (S1*S1+S2*S2+S3*S3).get_factor(0)/sigma
        W = Matrix([[(f*s).get_factor(0)/(2*volS) for s in twoforms] for f in curvatures])
        return W

    def get_metric_determinant(self):
        assert(len(self.tetrads)==4)
        vs = self.get_vectors()
        g_metric = Matrix([[Contract(self.metric*u*v,(0,2),(1,3)) for v in vs] for u in vs])
        return g_metric.det()
        
            

class VectorField():
    def __init__(self,manifold:Manifold,symbol):
        """
        Class: Vector Field

        This class represents a single term in a vector fields component expansion, however, it is purely symbolic so no basis is required.

        """
        self.symbol = symbol
        self.manifold = manifold
    
    def __eq__(self,other): return (self.symbol == other.symbol) and (self.manifold == other.manifold)
    def __hash(self): return hash((self.symbol,self.manfiold))

    def __mul__(self,other): return TensorProduct(self,other)
    def __rmul__(self,other): return TensorProduct(other,self)

    def __neg__(self):
        ret = Tensor()
        ret.comps_list = [[self]]
        ret.factors = [-1]
        return ret
    
    def __sub__(self,other): return self + (-other)
    def __rsub__(self,other): return (-self) + other
    
    def __add__(self,other):
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
    def __radd__(self,other): return self+other
        
    def _repr_latex_(self):
        return "$\\partial_{"+str(self.symbol)+"}$"

    def __str__(self): return "\\partial_{"+str(self.symbol)+"}"

    __repr__ = _repr_latex_
    _latex   = _repr_latex_
    _print   = _repr_latex_

class Tensor(): 
    def __init__(self, manifold:Manifold):
        self.__sympy__ = True
        self.manifold = manifold
        self.comps_list = []
        self.factors = []
    
    def __add__(self,other):
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
        ret.collect_comps()
        return ret
    
    def __radd__(self,other):
        return self + other

    def __sub__(self,other):
        return self + (-other)

    def __neg__(self):
        ret = Tensor(self.manifold)
        ret.comps_list = self.comps_list
        ret.factors = [-f for f in self.factors]
        return ret

    def __mul__(self,other):
        return TensorProduct(self,other)

    def __rmul__(self,other):
        return TensorProduct(other,self)

    def __div__(self,other): return TensorProduct(self,1/Number(other))
    def __truediv__(self,other): return TensorProduct(self,1/Number(other))

    def __getitem__(self,indices):
        s_weight = self.get_weight()
        if len(indices) != len(s_weight): raise IndexError("Indices not of correct form")
        if set(list(indices)) != set(range(len(s_weight))): raise IndexError("Each index must appear once and only once")
        if list(indices) == list(range(len(s_weight))): return self
        return PermuteIndices(self,list(indices))

    def _repr_latex_(self):
        latex_str = "$" + "+".join([ "(" + remove_latex_arguments(self.factors[i]) + ")" + r" \otimes ".join([str(f) for f in self.comps_list[i]]) for i in range(len(self.comps_list))])  + "$"
        if latex_str == "$$":
            return "$0$"
        return latex_str
    
    def is_vectorfield(self):
        for f in self.comps_list:
            if len(f) != 1 or not isinstance(f[0],VectorField):
                return False
        return True
    
    def get_weight(self):
        if len(self.factors) == 0: return (0,0)
        first_weight = tuple(map(lambda x: int(isinstance(x,VectorField))-int(isinstance(x,DifferentialForm)),self.comps_list[0]))
        for i in range(1,len(self.factors)):
            current_weight = tuple(map(lambda x: int(isinstance(x,VectorField))-int(isinstance(x,DifferentialForm)),self.comps_list[i]))
            if current_weight != first_weight: return (None)
        return first_weight
    
    def collect_comps(self):
        new_comps_list = []
        new_factors = []
        for i in range(len(self.comps_list)):
            if self.comps_list[i] not in new_comps_list:
                new_comps_list.append(self.comps_list[i])
                new_factors.append(self.factors[i])
            else:
                j = new_comps_list.index(self.comps_list[i])
                new_factors[j] += self.factors[i]
        
        i = 0
        while  i < len(new_comps_list):
            if new_factors[i] == 0:
                del new_factors[i]
                del new_comps_list[i]
                continue
            i+=1
    
        i = 0
        while i < len(new_comps_list):
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
        ret = Tensor(self.manifold)
        ret.comps_list = self.comps_list
        ret.factors = [simplify(f) for f in self.factors]
        ret.collect_comps()
        return ret

    def subs(self,target,sub=None,simp=True):
        ret = Tensor(self.manifold)
        ret.factors = self.factors
        ret.comps_list = self.comps_list

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
        
        if simp: ret = simplify(ret)
        return ret

    def get_factor(self,index):
        if len(self.factors) == 0: return 0
        return self.factors[index]
    
    def simplify(self):
        ret = Tensor(self.manifold)
        ret.factors = [f.simplify() for f in self.factors]
        ret.comps_list = self.comps_list
        ret.collect_comps()
        return ret

    def factor(self):
        ret = Tensor(self.manifold)
        ret.factors = [f.factor() for f in self.factors]
        ret.comps_list = self.comps_list
        ret.collect_comps()
        return ret

    def conjugate(self):
        ret = Tensor(self.manifold)
        ret.factors = [f.conjugate() for f in self.factors]
        ret.comps_list = self.comps_list
        ret.collect_comps()
        return ret

    def to_differentialform(self):
        if set(self.get_weight()) != set([-1]): raise TypeError("Tensor cannot be projected to a differential form")
        ret = DifferentialFormMul(self.manifold)
        ret.factors = [f/Number(factorial(len(self.factors))) for f in self.factors]
        ret.forms_list = self.comps_list.copy()

        ret.remove_squares()
        ret.remove_above_top()
        ret.sort_form_sums()
        ret.collect_forms()

        if ret.factors == [] and ret.forms_list == []: 
            return Number(0)

        return ret

    _sympystr = _repr_latex_
    __repr__  = _repr_latex_
    _latex    = _repr_latex_
    _print    = _repr_latex_

class DifferentialForm():
    def __init__(self,manifold,symbol,degree=0, exact=False):
        """
        Class: Differential Form

        This is the "atom" of a differential form in this package. It holds all the information needed for a generic differential form.
        
        """
        self.manifold = manifold
        self.degree = degree
        self.symbol = symbol
        self.exact = exact
        if degree < 0 or degree > self.manifold.dimension:
            self.symbol = Rational(0)
        
    def __eq__(self,other): return (self.symbol == other.symbol) and (self.degree == other.degree)
    def __hash__(self): return hash((str(self.symbol),self.degree))

    def __mul__(self,other): 
        if isinstance(other,(Tensor,VectorField)):
            return TensorProduct(self,other)
        return WedgeProduct(self,other)

    def __rmul__(self,other): 
        if isinstance(other,(Tensor,VectorField)):
            return TensorProduct(other,self)
        return WedgeProduct(other,self)
    def __div__(self,other): return WedgeProduct(self,1/other)
    def __truediv__(self,other): return WedgeProduct(self,1/other)

    def __add__(self,other):
        ret = DifferentialFormMul(self.manifold)
        ret.forms_list = [[self]]
        ret.factors = [[1]]
        if isinstance(other,(AtomicExpr,float,int,Number)):
            if other != 0:
                ret.forms_list.append([])
                ret.factors.append(other)
        elif isinstance(other,DifferentialForm):
            ret.forms_list.append([other])
            ret.factors.append(1)
        elif isinstance(other,DifferentialFormMul):
            ret.forms_list += other.forms_list
            ret.factors += other.factors
        else:
            raise NotImplementedError
        ret.collect_forms()
        return ret
    
    def __lt__(self,other):
        if not isinstance(other,DifferentialForm): raise NotImplementedError
        if str(self.symbol) < str(other.symbol):
            return True
        elif str(self.symbol) > str(other.symbol):
            return False
        else:
            return (self.degree) < other.degree

    def __neg__(self): return DifferentialFormMul(self.manifold,self,-1)
    def __sub__(self,other): return self + (-other)
    def __rsub__(self,other): return (-self) + other
    def __radd__(self,other): return self + other

    def __str__(self): return latex(self.symbol)

    def _repr_latex_(self): return self.symbol._repr_latex_()

    def __hash__(self): return hash((self.symbol,self.degree))

    def to_tensor(self): return (Number(1)*self).to_tensor()
    
    __repr__ = _repr_latex_
    _latex   = _repr_latex_
    _print   = _repr_latex_
    
    def __eq__(self,other):
        if isinstance(other,DifferentialForm):
            return str(self.symbol) == str(other.symbol) and self.degree == other.degree

    def _eval_simplify(self, **kwargs):
        return self
    
    def insert(self,vector:VectorField):
        if isinstance(vector,VectorField):
            if self.symbol == vector.symbol or str(self.symbol) == "d\\left("+str(vector.symbol)+"\\right)": return 1
            else: return Number(0)
        elif isinstance(vector,Tensor):
            if vector.is_vectorfield():
                return sum([vector.factors[i]*self.insert(vector.comps_list[i][0]) for i in range(len(vector.factors))])
        else:
            raise NotImplementedError

    def __getitem__(self,index):
        if len(self.factors) == 0: return 0
        return self.factors[index]

    @property
    def d(self):
        if self.exact: return DifferentialForm(self.manifold,Number(0),self.degree+1,exact=True)
        elif isinstance(self.symbol,Number): return DifferentialForm(self.manifold,Number(0),self.degree+1,exact=True)
        else:
            dsymbol = symbols(r"d\left("+str(self.symbol)+r"\right)")
            return DifferentialForm(self.manifold,dsymbol,degree=self.degree+1,exact=True)
        raise NotImplementedError

    def subs(self,target,sub=None):
        if target == self: return sub
        elif isinstance(target,DifferentialFormMul):
            if len(target.factors) == 1 and target.forms_list == [[self]]:
                return sub/target.factors[0]
        elif isinstance(target,dict):
            ret = DifferentialForm(self.symbol,self.degree)
            ret.exact = self.exact
            for t in target:
                ret = ret.subs(t,target[t])
            return ret
        else:
            ret = DifferentialForm(self.symbol,self.degree)
            ret.exact = self.exact
            return ret

class DifferentialFormMul():
    def __init__(self,manifold:Manifold,form:DifferentialForm=None,factor:AtomicExpr=None):
        self.__sympy__ = True
        if form == None:
            self.forms_list = []
            self.factors = []
        else:
            self.forms_list = [[form]]
            self.factors = [factor]
        self.manifold = manifold
 
    def __add__(self,other):
        ret = DifferentialFormMul(self.manifold)
        ret.forms_list = self.forms_list
        ret.factors = self.factors
        if isinstance(other,DifferentialFormMul):
            assert(self.manifold == other.manifold)
            ret.forms_list += other.forms_list
            ret.factors += other.factors
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
        if isinstance(other,(Tensor,VectorField)):
            return TensorProduct(self,other)
        return WedgeProduct(self,other)
    
    def __rmul__(self,other): 
        if isinstance(other,(Tensor,VectorField)):
            return TensorProduct(other,self)
        return WedgeProduct(other,self)

    def __div__(self,other): return WedgeProduct(self,(1/other))
    def __truediv__(self,other): return WedgeProduct(self,(1/other))

    def __radd__(self,other): return self + other
    def __neg__(self):
        ret = DifferentialFormMul(self.manifold)
        ret.forms_list = self.forms_list
        ret.factors = [-f for f in self.factors]
        return ret
    
    def __sub__(self,other): return self + (-other)
    def __rsub__(self,other): return other + (-self)

    def __eq__(self,other):
        if isinstance(other,DifferentialForm) and self.factors == [1] and len(self.forms_list[0]) == 1: return other == self.forms_list[0][0]
        elif not isinstance(other,DifferentialFormMul): raise NotImplementedError
        elif other.factors != self.factors: return False
        elif other.forms_list != self.forms_list: return False
        return True

    def __hash__(self): 
        symbols = []
        for forms in self.forms_list: symbols+=forms
        symbols += self.factors
        return hash(tuple(symbols))

    def insert(self,other):
        if isinstance(other,VectorField):
            ret = DifferentialFormMul(self.manifold)
            for i in range(len(self.forms_list)):
                sign = 1
                for j in range(len(self.forms_list[i])):
                    if self.forms_list[i][j].insert(other) != 0:
                        ret.forms_list += [self.forms_list[i][:j] + self.forms_list[i][j+1:]]
                        ret.factors += [self.factors[i]*sign]
                        break
                    sign *= (-1)**self.forms_list[i][j].degree 
        elif isinstance(other,Tensor) and other.is_vectorfield():
            ret = sum([other.factors[i]*self.insert(other.comps_list[i][0]) for i in range(len(other.factors))])
            return 
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
        i = 0
        while i < len(self.forms_list):
            deled = False
            for j in range(len(self.forms_list[i])):
                f = self.forms_list[i][j]
                if f.degree%2 == 1 and self.forms_list[i].count(f) > 1:
                    del self.forms_list[i]
                    del self.factors[i]
                    deled = True
                    break
            if not deled: i+=1
        
    def remove_above_top(self):
        i = 0
        while i < len(self.forms_list):
            if sum([f.degree for f in self.forms_list[i]]) > self.manifold.dimension:
                del self.forms_list[i]
                del self.factors[i]
                continue
            i += 1

    def sort_form_sums(self):
        for i in range(len(self.forms_list)):
            bubble_factor = 1
            for j in range(len(self.forms_list[i])):
                for k in range(j,len(self.forms_list[i])):
                    if self.forms_list[i][j] > self.forms_list[i][k]:
                        temp = self.forms_list[i][j]
                        self.forms_list[i][j] = self.forms_list[i][k]
                        self.forms_list[i][k] = temp
                        bubble_factor *= (-1)**(self.forms_list[i][j].degree*self.forms_list[i][k].degree)
            self.factors[i] = self.factors[i]*bubble_factor
    
    def collect_forms(self):
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
        latex_str = "$" + "+".join([ "(" + remove_latex_arguments(self.factors[i]) + ")" + r" \wedge ".join([str(f) for f in self.forms_list[i]]) for i in range(len(self.forms_list))]) + "$"
        if latex_str == "$$":
            return "$0$"
        return latex_str

    def get_factor(self,index):
        if len(self.factors) == 0: return 0
        return self.factors[index]
        
    def conjugate(self):
        ret = DifferentialFormMul(self.manifold)
        ret.factors = [f.conjugate() for f in self.factors]
        ret.forms_list = self.forms_list
        return ret

    def __str__(self):
        str_str = "+".join([ "(" + str(self.factors[i]) + ")" + r" \wedge ".join([str(f) for f in self.forms_list[i]]) for i in range(len(self.forms_list))])
        if str_str == "":
            return "0"
        return str_str

    def __getitem__(self,index):
        #TODO: Make this permute the differential forms not index the components
        pass
    _sympystr = _repr_latex_

    @property
    def d(self):
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
                d_factor = (-1)**sum([0] + [f.degree for f in self.forms_list[i][0:j]])
                new_forms_list += [self.forms_list[i][0:j] + [self.forms_list[i][j].d] + self.forms_list[i][j+1:]]
                new_factors_list += [d_factor*self.factors[i]]

        ret.forms_list = new_forms_list
        ret.factors = new_factors_list

        ret.remove_squares()
        ret.remove_above_top()
        ret.sort_form_sums()
        ret.collect_forms()

        return ret

    def _eval_simplify(self, **kwargs):
        ret = DifferentialFormMul(self.manifold)
        ret.forms_list = self.forms_list.copy()
        ret.factors = [simplify(f) for f in self.factors]
        
        ret.remove_squares()
        ret.remove_above_top()
        ret.sort_form_sums()
        ret.collect_forms()

        return ret
    
    def subs(self,target,sub=None):
        ret = DifferentialFormMul(self.manifold)
        ret.factors = self.factors
        ret.forms_list = self.forms_list

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
        return ret

    def to_tensor(self):
        ret = Tensor(self.manifold)
        for i in range(len(self.factors)):
            L = len(self.forms_list[i])
            for perm in permutations(list(range(L)),L):
                parity = (len(Permutation(perm).full_cyclic_form)-1)%2
                ret.comps_list += [[self.forms_list[i][p] for p in perm]]
                ret.factors += [(-1)**(parity+1)*self.factors[i]/factorial(L)]
        return ret

    def get_degree(self):
        weights = [sum(map(lambda x: x.degree,f)) for f in self.forms_list]
        if len(set(weights)) == 1:
            return weights[0]
        return None

    def get_component_at_basis(self,basis=None):
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
        return self._eval_simplify()

    def factor(self):
        ret = DifferentialFormMul(self.manifold)
        ret.forms_list = self.forms_list.copy()
        ret.factors = [simplify(f) for f in self.factors]
        
        ret.remove_squares()
        ret.remove_above_top()
        ret.sort_form_sums()
        ret.collect_forms()

        return ret    

    def expand(self):
        ret = DifferentialFormMul(self.manifold)
        ret.factors = [f.expand() for f in self.factors]
        ret.forms_list = self.forms_list
        return ret

def remove_latex_arguments(object):
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
    latex_str = remove_latex_arguments(object)
    display(Math(latex_str))

def DefScalars(names:str,**args)->Symbol:
    return symbols(names,**args)

def DefDifferentialForms(manifold:Manifold,symbs:list,degrees:list):
    ret = None
    if isinstance(symbs,str):
        ret = DefDifferentialForms(manifold,list(symbols(symbs)),degrees)
    elif isinstance(symbs,list):
        if isinstance(degrees,list):
            assert(len(symbs) == len(degrees))
            ret = [DifferentialForm(manifold,symbs[i],degrees[i]) for i in range(len(degrees))]
        elif isinstance(degrees,int):
            ret = [DifferentialForm(manifold,s,degrees) for s in symbs]
    else:
        raise NotImplementedError
    if isinstance(len,list) and len(ret) == 1:
        return ret[0]
    return ret

def DefVectorFields(manifold:Manifold,symbs:list):
    ret = None
    if isinstance(symbs,str):
        ret = DefVectorFields(manifold,list(symbols(symbs)))
    elif isinstance(symbs,(list,tuple)):
        ret = [VectorField(manifold,s) for s in symbs]
    else:
        raise NotImplementedError
    if len(ret) == 1:
        return ret[0]
    return ret

def DefConstants(names:str, **assumptions)->symbols:
    """ Uses the Quantity function to create constant symbols. """
    names = re.sub(r'[\s]+', ' ', names)
    constants = [Quantity(c,**assumptions) for c in names.split(' ') if c != '']
    if len(constants) == 1: return constants[0]
    return constants

def d(form,manifold=None):
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
        return ret

    raise NotImplementedError

def PartialDerivative(tensor,manifold=None):
    if isinstance(tensor,(DifferentialForm,DifferentialFormMul)):
        return PartialDerivative((1*tensor).to_tensor(),manifold)
    elif isinstance(tensor,(VectorField)):
        return 0
    elif isinstance(tensor,(AtomicExpr,Expr,Function)):
        if manifold == None: raise NotImplementedError("Manifold cannot be None for Scalar input")
        ret = Tensor(manifold)
        for i in range(manifold.dimension):
            ret.comps_list += [[manfiold.basis[i]]]
            ret.factors += [tensor.diff(manifold.coords[i])]
        ret.collect_comps()
        return ret
    elif isinstance(tensor,Tensor):
        ret = Tensor(tensor.manifold)
        man = tensor.manifold
        for i in range(man.dimension):
            for j in range(len(tensor.factors)):
                ret.comps_list += [[man.basis[i]]+tensor.comps_list[j]]
                ret.factors += [tensor.factors[j].diff(man.coords[i])]
        ret.collect_comps()
        return ret

def CovariantDerivative(tensor,manifold=None):
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
        ret.collect_comps()
        return ret
    elif isinstance(tensor,Tensor):
        t_weight = tensor.get_weight()
        Gamma = tensor.manifold.christoffel_symbols
        Gamma_tensor = Gamma*tensor
        CD_tensor = PartialDerivative(tensor)
        for i in range(len(t_weight)):
            if t_weight[i] == -1:
                # print([0,i+1]+[j for j in  list(range(1,len(t_weight)+1)) if j != i+1])
                index_list = [0] + list(range(2,len(t_weight)+1))
                index_list.insert(i+1,1)
                CD_tensor += -Contract(Gamma_tensor,(0,3+i))[*index_list]
            elif t_weight[i] == 1:
                index_list = list(range(1,len(t_weight)+1))
                index_list.insert(i+1,0)
                CD_tensor += Contract(Gamma_tensor,(2,3+i))[index_list]
        return CD_tensor

def WedgeProduct(left,right,debug=False):
    ret = None
    if isinstance(left,(int,float,Number,AtomicExpr,Expr)):
        if isinstance(right,(int,float,Number,AtomicExpr,Expr)):
            return left*right
        elif isinstance(right,DifferentialForm):
            ret = DifferentialFormMul(right.manifold)
            ret.forms_list = [[right]]
            ret.factors = [left]
        elif isinstance(right,DifferentialFormMul):
            ret = DifferentialFormMul(right.manifold)
            ret.forms_list = right.forms_list
            ret.factors = [left*f for f in right.factors]
        else:
            raise NotImplementedError
    elif isinstance(left, DifferentialForm):
        ret = DifferentialFormMul(left.manifold)
        if isinstance(right,(int,float,Number,AtomicExpr,Expr)):
            ret.forms_list = [[left]]
            ret.factors = [right]
        elif isinstance(right,DifferentialForm):
            assert(right.manifold == left.manifold)
            ret.forms_list = [[left,right]]
            ret.factors = [1]
        elif isinstance(right,DifferentialFormMul):
            assert(right.manifold == left.manifold)
            ret.forms_list = [[left]+rf for rf in right.forms_list]
            ret.factors = right.factors
        else:
            raise NotImplementedError
    elif isinstance(left,DifferentialFormMul):
        ret = DifferentialFormMul(left.manifold)
        if isinstance(right,(int,float,Number,AtomicExpr,Expr)):
            ret.forms_list = left.forms_list
            ret.factors = [right*f for f in left.factors]
        elif isinstance(right,DifferentialForm):
            assert(left.manifold == right.manifold)
            ret.forms_list = [lf+[right] for lf in left.forms_list]
            ret.factors = left.factors
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
    if isinstance(left,DifferentialFormMul) or isinstance(right,DifferentialFormMul): raise NotImplementedError("Must convert DifferentialFormMul into Tensor before using with TensorProduct")
    ret = None
    if isinstance(left,(int,float,AtomicExpr,Expr)):
        if isinstance(right,(int,float,AtomicExpr,Expr)):
            return left*right
        elif isinstance(right,(DifferentialForm,VectorField)):
            ret = Tensor(right.manifold)
            ret.comps_list = [[right]]
            ret.factors = [left]
        elif isinstance(right,Tensor):
            ret = Tensor(right.manifold)
            ret.comps_list = right.comps_list
            ret.factors = [left*f for f in right.factors]
        else:
            raise NotImplementedError
    elif isinstance(left,VectorField):
        ret = Tensor(left.manifold)
        if isinstance(right,(int,float,AtomicExpr,Expr)):
            ret.comps_list = [[left]]
            ret.factors = [right]
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
            ret.comps_list = [[left]]
            ret.factors = [right]
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
            ret.comps_list = left.comps_list
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
    ret.collect_comps()
    return ret

def Contract(tensor,*positions):
    if not isinstance(tensor,Tensor): raise TypeError("First argument must be a Tensor.")
    elif isinstance(tensor,(int,float,Number)): return tensor
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
        if isinstance(sign,DifferentialFormMul):
            display(sign.forms_list)
            display(sign.factors)
        if sign != 0:
            ret.comps_list += [total_without]
            ret.factors += [tensor.factors[i]*sign]
    ret.collect_comps()
    if ret.comps_list ==[[]]: return ret.factors[0]
    if ret.comps_list == []: return Number(0)
    return ret

def PermuteIndices(tensor,new_order):
    if isinstance(tensor,(int,float,Number)): return tensor
    t_weight = tensor.get_weight()
    if (len(new_order)!=len(t_weight)): raise NotImplementedError("New index order must contain every index")
    if set(new_order) != set(range(len(t_weight))): raise TypeError("New index order does not contain every index once and only once")
    ret = Tensor(tensor.manifold)
    for i in range(len(tensor.factors)):
        ret.factors += [tensor.factors[i]]
        ret.comps_list += [[tensor.comps_list[i][j] for j in new_order]]
    
    ret.collect_comps()
    return ret

def LieDerivative(vector,tensor):
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
                LieD_tensor += sign*Contract(tensor*PDvector,(i,0 if sign == 1 else 1))
        return LieD_tensor
    raise NotImplementedError("Only the Tensor class and the Differential form class can be acted on by the LieDerivative")

def FormsListInBasisMatrix(formslist:dict, basis=None) -> Matrix:
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
    
def SO3TwoFormProduct(Si,Bi,g_UU):
    S1_DD,S2_DD,S3_DD = [2*s.to_tensor() for s in Si]
    B1_DD,B2_DD,B3_DD = [2*b.to_tensor() for b in Bi]
    S1_DU,S2_DU,S3_DU = [Contract(s_DD*g_UU,(1,2)) for s_DD in [S1_DD,S2_DD,S3_DD]]
    R1_DD = Contract(S2_DU*B3_DD - S3_DU*B2_DD,(1,2))
    R2_DD = Contract(S3_DU*B1_DD - S1_DU*B3_DD,(1,2))
    R3_DD = Contract(S1_DU*B2_DD - S2_DU*B1_DD,(1,2))

    R1 = R1_DD.to_differentialform()/Number(2)
    R2 = R2_DD.to_differentialform()/Number(2)
    R3 = R3_DD.to_differentialform()/Number(2)
    return [R1,R2,R3]

def Hodge(form):
    basis = form.manifold.basis
    dim = form.manifold.dimension
    degree = form.get_degree()

    for left_index in variations(range(dim),degree):
        left_index = list(left_index)
        right_index_list = [i for i in range(dim) if i not in left_index]
        display(left_index)
        for right_index in variations(right_index_list,3):
            display(right_index)


    # TODO: Implements Solver first
    # Allow basis to be generic 1-form and then solve for everything in terms of those
    # SetBasis([dt+dx,dt-dx,dy,dz]) and solve tmp_e0 = dt+dx, tmp_e1 = dt-dx and so on
    # Then Solve for dt,dx,dy,dz = f(tmp_eI) and substitute into imput and then compute HodgeStar given eI's
    # Use signature as well and then substitute back original dt,dx,dy,dz 
    pass
    # ret = DifferentialFormMul(form.manifold)
    # assert(form.manifold.basis != None)
    # basis = form.manifold.basis
    # signature = form.manifold.signature
    # full_index = list(range(len(basis)))
    # if isinstance(form,DifferentialFormMul):
    #     for i in range(len(form.factors)):
    #         term = form.forms_list[i]
    #         factor = form.factors[i]
    #         indices = [basis.index(t) for t in term if t in basis]
    #         new_indices = [idx for idx in full_index if idx not in indices]
    #         sign = LeviCivita(*(indices+new_indices))*signature**int(0 in indices)
    #         ret = ret + (sign*form.factors[i])*prod([basis[j] for j in new_indices])
    # elif isinstance(form,DifferentialForm):
    #     indices = [basis.index(form)]
    #     new_indices = [idx for idx in full_index if idx not in indices]
    #     sign = LeviCivita(*(indices+new_indices))*signature**int(0 in indices)
    #     ret = ret + sign*prod([basis[j] for j in new_indices])
    # elif isinstance(form,(int,float,Number,Expr)):
    #     return form*prod(basis)
    # else:
    #     raise NotImplementedError
    # return ret

    if weyl == None:
        weyl = SelfDualWeylMatrix(curvatures, twoforms)
    F1,F2,F3 = curvatures
    S1,S2,S3 = twoforms

    R1 = F1 - weyl[0,0]*S1 - weyl[0,1]*S2 - weyl[0,2]*S3
    R2 = F2 - weyl[1,0]*S1 - weyl[1,1]*S2 - weyl[1,2]*S3
    R3 = F3 - weyl[2,0]*S1 - weyl[2,1]*S2 - weyl[2,2]*S3

    return [R1,R2,R3]