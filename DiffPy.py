from sympy import Symbol, I, Integer, AtomicExpr, Rational, latex, Number, Expr

class DifferentialForm:
    def __init__(self,symbol,degree=0,max_degree=4, exact=False):
        assert degree >= 0
        assert degree <= max_degree
        self.max_degree = degree
        self.degree = degree
        self.symbol = symbol

    def __mul__(self,other):
        if isinstance(other,AtomicExpr):
            return DifferentialFormMul(self,other)
        elif isinstance(other,Expr):
            return DifferentialFormMul(self,other)
        elif isinstance(other,int):
            return DifferentialFormMul(self,Integer(other))
        elif isinstance(other,float):
            return DifferentialFormMul(self,Rational(other))
        elif isinstance(other,DifferentialForm):
            ret = DifferentialFormMul()
            ret.forms_list = [[self,other]]
            ret.factors = [1]
            return ret
        elif isinstance(other,DifferentialFormMul):
            return DifferentialFormMul(self,1)*other
        else:
            raise NotImplementedError("Not implemented multplication of type '"+str(type(self).__name__)+" * "+str(type(other).__name__)+"'")
    
    def __add__(self,other):
        if isinstance(other,AtomicExpr):
            return DifferentialForm(other,0)+self
        elif isinstance(other,Expr):
            return DifferentialForm(other,0)+self
        elif isinstance(other,int):
            return DifferentialForm(Integer(other),0)+self
        elif isinstance(other,float):
            return DifferentialForm(Rational(other),0)+self
        elif isinstance(other,DifferentialForm):
            return DifferentialFormMul(self,1)+other
        elif isinstance(other,DifferentialFormMul):
            ret = DifferentialFormMul()
            ret.forms_list = [[self]]+other.forms_list
            ret.factors = [1]+other.factors
            return ret
        else:
            raise NotImplementedError
        
    def __neg__(self): return DifferentialFormMul(self,-1)
    def __sub__(self,other): return self + (-other)
    def __rsub__(self,other): return (-self) + other
    def __radd__(self,other): return self.__add__(other)
    def __rmul__(self,other): return self * other

    def __str__(self):
        return latex(self.symbol)

    def _repr_latex_(self):
        return self.symbol._repr_latex_()

class DifferentialFormMul():
    def __init__(self,form:DifferentialForm=None,factor:AtomicExpr=None):
        if form == None:
            self.forms_list = []
            self.factors = []
        else:
            self.forms_list = [[form]]
            self.factors = [factor]
    
    def __add__(self,other):
        if isinstance(other,DifferentialFormMul):
            ret = DifferentialFormMul()
            ret.forms_list += (self.forms_list)
            ret.forms_list += (other.forms_list)
            ret.factors += self.factors + other.factors
            return ret
        elif isinstance(other,DifferentialForm):
            ret = DifferentialFormMul()
            ret.forms_list += self.forms_list
            ret.factors += self.factors
            if isinstance(other.symbol,Number):
                ret.forms_list += [[DifferentialForm(Number(1),0)]]
                ret.factors += [other.symbol]
            elif isinstance(other.symbol,Expr):
                ret.forms_list += [[DifferentialForm(Number(1),0)]]
                ret.factors += [other.symbol]
            elif isinstance(other.symbol,AtomicExpr):
                ret.forms_list += [[DifferentialForm(Number(1),0)]]
                ret.factors += [other.symbol]
            else:
                ret.forms_list += [[other]]
                ret.factors += [1]
            return ret
        elif isinstance(other,int):
            return self + DifferentialForm(Integer(other),0)
        elif isinstance(other,float):
            return self + DifferentialForm(Rational(other),0)
        elif isinstance(other,AtomicExpr):
            return self + DifferentialForm(other,0)
        elif isinstance(other,Expr):
            return self + DifferentialForm(other,0)
        else:
            raise NotImplementedError
    
    def __mul__(self,other):
        if isinstance(other,int):
            ret = DifferentialFormMul()
            ret.forms_list = self.forms_list
            ret.factors = [Integer(other)*f for f in self.factors]
            return ret
        elif isinstance(other,float):
            ret = DifferentialFormMul()
            ret.forms_list = self.forms_list
            ret.factors = [Rational(other)*f for f in self.factors]
            return ret

        elif isinstance(other,AtomicExpr):
            ret = DifferentialFormMul()
            ret.forms_list = self.forms_list
            ret.factors = [(other)*f for f in self.factors]
            return ret

        elif isinstance(other,Expr):
            ret = DifferentialFormMul()
            ret.forms_list = self.forms_list
            ret.factors = [(other)*f for f in self.factors]
            return ret            

        elif isinstance(other,DifferentialForm):
            ret = DifferentialFormMul()
            ret.forms_list = [fl+[other] for fl in self.forms_list]
            ret.factors = self.factors

            return ret
        elif isinstance(other,DifferentialFormMul):
            ret = DifferentialFormMul()
            for i in range(len(self.forms_list)):
                for j in range(len(other.forms_list)):
                    ret.forms_list.append(self.forms_list[i]+other.forms_list[j])
                    ret.factors.append(self.factors[i]*other.factors[j])
            
            return ret
        else:
            raise NotImplementedError

    def __radd__(self,other): return self + other
    def __neg__(self):
        ret = DifferentialFormMul()
        ret.forms_list = self.forms_list
        ret.factors = [-f for f in self.factors]
        return ret
    
    def __sub__(self,other): return self + (-other)
    def __rsub__(self,other): return other + (-self)

    def _repr_latex_(self):
        latex_str = "$" + "+".join([ "(" + latex(self.factors[i]) + ")" + r"\wedge".join([str(f) for f in self.forms_list[i]]) for i in range(len(self.forms_list))]) + "$"
        return latex_str
    

        
    