from sympy import Symbol, I, Integer, AtomicExpr, Rational, latex, Number, Expr, symbols, simplify, Function
from sympy.physics.units.quantities import Quantity
from IPython.display import Math
from sympy.combinatorics import Permutation
from itertools import permutations
import re
import numbers
from math import factorial

MAX_DEGREE = 4

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

def set_max_degree(max_degree: int):
    MAX_DEGREE=max_degree

def constants(names:str)->symbols:
    """ Uses the Quantity function to create constant symbols. """
    names = re.sub(r'[\s+]', ' ', names)
    return [Quantity(c) for c in names.split(' ')]

class VectorField():
    def __init__(self,symbol):
        """
        Class: Vector Field

        This class represents a single term in a vector fields component expansion, however, it is purely symbolic so no basis is required.

        """
        self.symbol = symbol
    
    def __eq__(self,other): return self.symbol == other.symbol
    def __hash(self): return hash(self.symbol)

    def __mull__(self,other):
        #TODO: Implement
        """
        Pseudo Code:

        IF other is a (VectorField or DifferentialForm):
            return TensorProduct(self,other)
        ELIF other is a (Tensor or DifferentialFormMul):
            return SUM([TensorProduct(self,term) term in other])
        """
        pass
    
    def __add__(self,other):
        #TODO: Implement
        pass

    def _repr_latex_(self):
        return "$\\partial_{"+str(self.symbol)+"}$"

    __repr__ = _repr_latex_
    _latex   = _repr_latex_
    _print   = _repr_latex_

class Tensor():
    def __init__(self):
        self.__sympy__ = True
        self.comps_list = []
        self.factors = []
    
    def __add__(self,other):
        ret = Tensor()
        ret.comps_list += self.comps_list
        ret.factors += self.factors
        if isinstance(other,Tensor):
            ret.comps_list +=  (other.comps_list)
            ret.factors += other.factors
        elif isinstance(other,DifferentialForm):
            if isinstance(other.symbol,Number):
                ret.comps_list += [[DifferentialForm(Number(1),0,exact=True)]]
                ret.factors += [other.symbol]
            elif isinstance(other.symbol,AtomicExpr):
                ret.comps_list += [[DifferentialForm(Number(1),0,exact=True)]]
                ret.factors += [other.symbol]
            else:
                ret.comps_list += [[other]]
                ret.factors += [Number(1)]
        elif isinstance(other,VectorField):
            ret.comps_list += [[other]]
            ret.factors += [Number(1)]
        elif isinstance(other,DifferentialFormMul):
            #TODO: Convert differential form to tensor and add it on
            pass
        elif isinstance(other,float) or isinstance(other,int):
            ret = self + DifferentialForm(Rational(other),0)
        elif isinstance(other,AtomicExpr):
            ret = self + DifferentialForm(other,0)
        else:
            raise NotImplementedError
        
    def __mull__(self,other):
        #TODO: Implement this probably through TensorProduct(?)
        pass

    def _repr_latex_(self):
        latex_str = "$" + "+".join([ "(" + remove_latex_arguments(self.factors[i]) + ")" + r" \otimes ".join([str(f) for f in self.comps_list[i]]) for i in range(len(self.comps_list))]) + "$"
        if latex_str == "$$":
            return "$0$"
        return latex_str
    
    _sympystr = _repr_latex_

class DifferentialForm():
    def __init__(self,symbol,degree=0, exact=False):
        """
        Class: Differential Form

        This is the basic class of this package. It holds all the information needed for a generic differential form.
        
        """
        self.degree = degree
        self.symbol = symbol
        self.exact = exact
        if degree < 0 or degree > MAX_DEGREE:
            self.symbol = Ratioanl(0)
        
    def __eq__(self,other): return (self.symbol == other.symbol) and (self.degree == other.degree)
    def __hash__(self): return hash((str(self.symbol),self.degree))

    def __mul__(self,other): return WedgeProduct(self,other)
    def __rmul__(self,other): return WedgeProduct(other,self)
    def __div__(self,other): return WedgeProduct(self,1/other)

    def __add__(self,other):
        ret = DifferentialFormMul()
        if isinstance(other,AtomicExpr) or isinstance(other,float) or isinstance(other,int):
            ret.forms_list = [[self],[DifferentialForm(Integer(1),0)]]
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

    def __str__(self):
        return latex(self.symbol)

    def _repr_latex_(self):
        return self.symbol._repr_latex_()
    
    __repr__ = _repr_latex_
    _latex   = _repr_latex_
    _print   = _repr_latex_
    
    def __eq__(self,other):
        if isinstance(other,DifferentialForm):
            return str(self.symbol) == str(other.symbol) and self.degree == other.degree
    
    def insert(self,vector:VectorField):
        if isinstance(vector,VectorField):
            if self.symbol == vector.symbol: return 1
            else: return 0
        else:
            raise NotImplementedError

    @property
    def d(self):
        if self.exact: return DifferentialForm(Number(0),self.degree+1,exact=True)
        elif isinstance(self.symbol,Number): return DifferentialForm(Number(0),self.degree+1,exact=True)
        else:
            dsymbol = symbols(r"d\left("+str(self.symbol)+r"\right)")
            return DifferentialForm(dsymbol,degree=self.degree+1,exact=True)
        raise NotImplementedError
    
    def _eval_simplify(self, **kwargs):
        return self

class DifferentialFormMul():

    def __init__(self,form:DifferentialForm=None,factor:AtomicExpr=None):
        self.__sympy__ = True
        if form == None:
            self.forms_list = []
            self.factors = []
        else:
            self.forms_list = [[form]]
            self.factors = [factor]
 
    def __add__(self,other):
        ret = DifferentialFormMul()
        if isinstance(other,DifferentialFormMul):
            ret.forms_list += (self.forms_list) + (other.forms_list)
            ret.factors += self.factors + other.factors
        elif isinstance(other,DifferentialForm):
            ret.forms_list += self.forms_list + [[other]]
            ret.factors += self.factors + [1]
        elif isinstance(other,(float,int)):
            ret = self + DifferentialForm(Rational(other),0)
        elif isinstance(other,AtomicExpr):
            ret = self + DifferentialForm(other,0)
        else:
            raise NotImplementedError
        ret = simplify(ret)

        return ret
    
    def __mul__(self,other): return WedgeProduct(self,other)
    
    def __rmul__(self,other): return WedgeProduct(other,self)

    def __div__(self,other): return WedgeProduct(self,(1/other))

    def __radd__(self,other): return self + other
    def __neg__(self):
        ret = DifferentialFormMul()
        ret.forms_list = self.forms_list
        ret.factors = [-f for f in self.factors]
        return ret
    
    def __sub__(self,other): return self + (-other)
    def __rsub__(self,other): return other + (-self)

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
            if sum([f.degree for f in self.forms_list[i]]) > MAX_DEGREE:
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
        ret = DifferentialFormMul()
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
                        print(ret.forms_list[i], sub)
                        print([ret.forms_list[i][:j] + [sub] + ret.forms_list[i][j+1:]])
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
            if len(target.factors) > 1: raise NotImplementedError("Cannot match more than 1 term at a time")
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
                            new_forms_list += [ret.forms_list[i][:match_index] + s + ret.forms_list[i][match_index+len(target.forms_list[0])+1:]]
                            new_factors_list.append(ret.factors[i]*f/target.factors[0])
                    elif isinstance(sub,DifferentialForm):
                        new_forms_list += [ret.forms_list[i][:match_index] + [sub] + ret.forms_list[i][match_index+len(target.forms_list[0]):]]
                        new_factors_list.append(ret.factors[i]/target.factors[0])
                    elif isinstance(sub,float) or isinstance(sub,int) or isinstance(sub,AtomicExpr):
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

        if not isinstance(sub,DifferentialForm) and not isinstance(sub,DifferentialFormMul) and sub != None:
            for i in range(len(self.factors)):
                    ret.factors[i] = ret.factors[i].subs(target,sub)

        ret = simplify(ret)
        return ret

    def insert(self,vect:VectorField):
        ret = DifferentialFormMul()
        if isinstance(vect,VectorField):
            for i in range(len(self.factors)):
                sign = 1
                for j in range(len(self.forms_list[i])):
                    if self.forms_list[i][j].symbol == vect.symbol:
                        ret.forms_list += [self.forms_list[i][:j] + self.forms_list[i][j+1:]]
                        ret.factors += [self.factors[i]*sign]
                        break
                    else:
                        sign *= (-1)**self.forms_list[i][j].degree
        else:
            raise NotImplementedError
        
        return ret

    def to_tensor(self):
        ret = Tensor()
        for i in range(len(self.factors)):
            L = len(self.forms_list[i])
            for perm in permutations(list(range(L)),L):
                parity = (len(Permutation(perm).full_cyclic_form)-1)%2
                ret.comps_list += [[self.forms_list[i][p] for p in perm]]
                ret.factors += [(-1)**parity*self.factors[i]/factorial]
        return ret


def d(form):
    if isinstance(form,(DifferentialForm,DifferentialFormMul)):
        return form.d
    
    elif isinstance(form,AtomicExpr):
        ret = DifferentialFormMul()
        new_forms_list = []
        new_factors_list = []
        for f in form.free_symbols:
            dform = form.diff(f)
            if dform != 0:
                new_forms_list += [[DifferentialForm(f,0).d]]
                new_factors_list += [dform]
        
        ret.forms_list = new_forms_list
        ret.factors = new_factors_list
        return ret

    raise NotImplementedError

def WedgeProduct(left,right):
    ret = DifferentialFormMul()
    if isinstance(left,(int,float,Number)):
        if isinstance(right,(int,float,Number)):
            return left*right
        elif isinstance(right,DifferentialForm):
            ret.forms_list = [[right]]
            ret.factors = [left]
        elif isinstance(right,DifferentialFormMul):
            ret.forms_list = right.forms_list
            ret.factors = [left*f for f in right.factors]
        else:
            raise NotImplementedError
    elif isinstance(left, DifferentialForm):
        if isinstance(right,(int,float,Number)):
            ret.forms_list = [[left]]
            ret.factors = [right]
        elif isinstance(right,DifferentialForm):
            ret.forms_list = [[left,right]]
            ret.factors = [1]
        elif isinstance(right,DifferentialFormMul):
            ret.forms_list = [[left]+rf for rf in right.forms_list]
            ret.factors = right.factors
        else:
            raise NotImplementedError
    elif isinstance(left,DifferentialFormMul):
        if isinstance(right,(int,float,Number)):
            ret.forms_list = left.forms_list
            ret.factors = [right*f for f in left.factors]
        elif isinstance(right,DifferentialForm):
            ret.forms_list = [lf+[right] for lf in left.forms_list]
            ret.factors = left.factors
        elif isinstance(right,DifferentialFormMul):
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
    ret = Tensor()
    # TODO: Implement in similar way to WedgeProduct is implemented