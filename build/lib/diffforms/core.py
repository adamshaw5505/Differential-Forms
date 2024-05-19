from sympy import Symbol, I, Integer, AtomicExpr, Rational, latex, Number, Expr, symbols, simplify, Function, LeviCivita
from sympy.physics.units.quantities import Quantity
from IPython.display import Math
from sympy.combinatorics import Permutation
from itertools import permutations
import re
import numbers
from math import factorial, prod

class Manifold():
    def __init__(self,label,dimension):
        self.label = label
        assert(dimension > 0)
        self.dimension = dimension
        self.signature = None
        self.basis = None
    
    def __eq__(self,other):
        if isinstance(other,Manifold):
            return (self.label == other.label) and (self.dimension == other.dimension)
        return False
    
    def __len__(self): return self.dimension

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
        if isinstance(self,(int,float,AtomicExpr,Expr)):
            ret.comps_list = [[self],[1]]
            ret.factors = [1,1]
        elif isinstance(other,(VectorField,DifferentialForm)):
            ret.comps_list = [[self],[other]]
            ret.factors = [1,1]
        elif isinstance(other,DifferentialFormMul):
            return self + other.to_tensor()
        elif isinstance(other,Tensor):
            ret.comps_list = [[self]] + other.comps_list
            ret.factors = [1] + other.factors
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
        ret.comps_list += self.comps_list
        ret.factors += self.factors
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
            ret = self + DifferentialForm(Rational(other),0)
        elif isinstance(other,AtomicExpr):
            ret = self + DifferentialForm(other,0)
        else:
            raise NotImplementedError
        ret.collect_comps()
        return ret

    def __sub__(self,other):
        return self + (-other)

    def __neg__(self):
        ret = Tensor(self.manifold)
        ret.comps_list = self.comps_list
        ret.factors = [-f for f in self.factors]
        return ret

    def __mull__(self,other):
        return TensorProduct(self,other)

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

    def subs(self,target,subs=None):
        #TODO: Implement subs function
        pass

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
        if isinstance(other,AtomicExpr) or isinstance(other,float) or isinstance(other,int):
            ret.forms_list = [[self],[DifferentialForm(self.manifold,Integer(1),0)]]
            ret.factors = [1,other]
        elif isinstance(other,DifferentialForm):
            ret.forms_list = [[self],[other]]
            ret.factors = [1,1]
        elif isinstance(other,DifferentialFormMul):
            ret.forms_list = [[self]]+other.forms_list
            ret.factors = [1]+other.factors
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

    def __neg__(self): return DifferentialFormMul(self,-1)
    def __sub__(self,other): return self + (-other)
    def __rsub__(self,other): return (-self) + other
    def __radd__(self,other): return self + other

    def __str__(self): return latex(self.symbol)

    def _repr_latex_(self): return self.symbol._repr_latex_()

    def __hash__(self): return hash((self.symbol,self.degree))
    
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
            else: return 0
        elif isinstance(vector,Tensor):
            if vector.is_vectorfield():
                return sum([vector.factors[i]*self.insert(vector.comps_list[i][0]) for i in range(len(vector.factors))])
        else:
            raise NotImplementedError

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
        if isinstance(other,DifferentialFormMul):
            assert(self.manifold == other.manifold)
            ret.forms_list += (self.forms_list) + (other.forms_list)
            ret.factors += self.factors + other.factors
        elif isinstance(other,DifferentialForm):
            assert(self.manifold == other.manifold)
            ret.forms_list += self.forms_list + [[other]]
            ret.factors += self.factors + [1]
        elif isinstance(other,(float,int)):
            ret = self + DifferentialForm(self.manifold,Rational(other),0)
        elif isinstance(other,AtomicExpr):
            ret = self + DifferentialForm(self.manifold,other,0)
        else:
            raise NotImplementedError
        ret = simplify(ret)
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
        ret = DifferentialFormMul()
        ret.forms_list = self.forms_list
        ret.factors = [-f for f in self.factors]
        return ret
    
    def __sub__(self,other): return self + (-other)
    def __rsub__(self,other): return other + (-self)

    def __eq__(self,other):
        if isinstance(other,DifferentialForm) and self.factors == [1] and len(self.forms_list[0]) == 1: return other == self.forms_list[0][0]
        elif not isinstance(DifferentialFormMul): raise NotImplementedError
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
            return ret
        elif isinstance(other,Tensor) and other.is_vectorfield():
            return sum([other.factors[i]*self.insert(other.comps_list[i][0]) for i in range(len(other.factors))])
        else:
            raise NotImplementedError

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

    def __str__(self):
        str_str = "+".join([ "(" + str(self.factors[i]) + ")" + r" \wedge ".join([str(f) for f in self.forms_list[i]]) for i in range(len(self.forms_list))])
        if str_str == "":
            return "0"
        return str_str
    
    _sympystr = _repr_latex_

    @property
    def d(self):
        ret = DifferentialFormMul()
        new_forms_list = []
        new_factors_list = []
        for i in range(len(self.forms_list)):
            fact = self.factors[i]
            if hasattr(fact,"free_symbols"):
                for f in fact.free_symbols:
                    dfact = fact.diff(f)
                    if dfact != 0:
                        new_forms_list += [[DifferentialForm(f,0).d] + self.forms_list[i]]
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
        ret.factors = []
        for i in range(len(self.factors)):
            ret.factors.append(simplify(self.factors[i]))
        
        ret.remove_squares()
        ret.remove_above_top()
        ret.sort_form_sums()
        ret.collect_forms()

        return ret
    
    def subs(self,target,sub=None):
        ret = DifferentialFormMul()
        ret.factors = self.factors
        ret.forms_list = self.forms_list

        if isinstance(target,DifferentialForm):
            new_forms_list = []
            new_factors_list = []
            for i in range(len(ret.forms_list)):
                if target in ret.forms_list[i]:
                    j = ret.forms_list[i].index(target)
                    if isinstance(sub,DifferentialForm):
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
        
        ret = simplify(ret)
        return ret

    def to_tensor(self):
        ret = Tensor(self.manifold)
        for i in range(len(self.factors)):
            L = len(self.forms_list[i])
            for perm in permutations(list(range(L)),L):
                parity = (len(Permutation(perm).full_cyclic_form)-1)%2
                ret.comps_list += [[self.forms_list[i][p] for p in perm]]
                ret.factors += [(-1)**parity*self.factors[i]/factorial(L)]
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
    return ret

def DefVectorFields(manifold:Manifold,symbs:list):
    ret = None
    if isinstance(symbs,str):
        ret = DefVectorFields(manifold,list(symbols(symbs)))
    elif isinstance(symbs,list):
        ret = [VectorField(manifold,s) for s in symbs]
    else:
        raise NotImplementedError
    return ret

def DefConstants(names:str)->symbols:
    """ Uses the Quantity function to create constant symbols. """
    names = re.sub(r'[\s]+', ' ', names)
    return [Quantity(c) for c in names.split(' ')]

def SetBasis(manifold:Manifold,basis:list,signature=1):
    assert(len(basis) == manifold.dimension)
    manifold.basis = basis
    manifold.signature = signature

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

def WedgeProduct(left,right):
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
    
    ret = simplify(ret)
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
                    ret.comps_list += [left.comps_list[i]+right.coms_list[j]]
                    ret.factors += [left.factors[i]*right.factors[j]]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    ret.collect_comps()
    return ret

def Contract(left,right,*positions):
    if not (isinstance(left,Tensor) and isinstance(right,Tensor)): raise TypeError("First two arguments must be Tensor.")
    assert(left.manifold == right.manifold)
    left_weight = left.get_weight()
    right_weight = right.get_weight()
    if left_weight == (None) or right_weight == (None): raise TypeError("Tensors must be of consistent types")
    p1_list = []
    p2_list = []
    for p in positions:
        p1,p2 = p
        p1_list += [p1]
        p2_list += [p2]
        if p1 > len(left_weight) or p2 > len(right_weight) or p1 < 0 or p2 < 0: raise IndexError("Contraction index out of range.")
        if left_weight[p1]*right_weight[p2] == 1: raise NotImplementedError("Tensor Contraction must be between vector fields and differential forms.")
    ret = Tensor(left.manifold)
    for i in range(len(left.factors)):
        for j in range(len(right.factors)):
            left_poped = [e for k,e in enumerate(left.comps_list[i]) if k in p1_list]
            right_poped = [e for k,e in enumerate(right.comps_list[j]) if k in p2_list]
            left_without = [e for k,e in enumerate(left.comps_list[i]) if k not in p1_list]
            right_without = [e for k,e in enumerate(right.comps_list[j]) if k not in p2_list]
            sign = 1
            for k in range(len(left_poped)):
                if isinstance(left_poped[k],DifferentialForm):
                    sign *= left_poped[k].insert(right_poped[k])
                else:
                    sign *= right_poped[k].insert(left_poped[k])
            if sign != 0:
                ret.comps_list += [left_without + right_without]
                ret.factors += [left.factors[i]*right.factors[j]]
    ret.collect_comps()
    return ret
    
def Hodge(form):
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