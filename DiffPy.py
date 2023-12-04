from sympy import Symbol, I, Integer, AtomicExpr, Rational, latex, Number, Expr

class DifferentialForm:
    def __init__(self,symbol,degree=0,max_degree=4, exact=False):
        assert degree >= 0
        assert degree <= max_degree
        self.max_degree = degree
        self.degree = degree
        self.symbol = symbol
        self.exact = exact

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
        ret = DifferentialFormMul()
        if isinstance(other,AtomicExpr):
            ret.forms_list = [[self],[DifferentialForm(Integer(1),0)]]
            ret.factors = [1,other]
        elif isinstance(other,Expr):
            ret.forms_list = [[self],[DifferentialForm(Integer(1),0)]]
            ret.factors = [1,other]
        elif isinstance(other,int):
            ret.forms_list = [[self],[DifferentialForm(Integer(1),0)]]
            ret.factors = [1,other]
        elif isinstance(other,float):
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
            self.max_degree = 4
        else:
            self.forms_list = [[form]]
            self.factors = [factor]
            self.max_degree = form.max_degree
    
    def __add__(self,other):
        ret = DifferentialFormMul()
        if isinstance(other,DifferentialFormMul):
            ret.forms_list += (self.forms_list)
            ret.forms_list += (other.forms_list)
            ret.factors += self.factors + other.factors

            ret.__sort_form_sums()
            ret.__collect_forms()
            return ret
        elif isinstance(other,DifferentialForm):
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
            ret.__collect_forms()
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
        ret = DifferentialFormMul()
        if isinstance(other,int):
            ret.forms_list = self.forms_list
            ret.factors = [Integer(other)*f for f in self.factors]

        elif isinstance(other,float):
            ret.forms_list = self.forms_list
            ret.factors = [Rational(other)*f for f in self.factors]

        elif isinstance(other,AtomicExpr):
            ret.forms_list = self.forms_list
            ret.factors = [(other)*f for f in self.factors]

        elif isinstance(other,Expr):
            ret.forms_list = self.forms_list
            ret.factors = [(other)*f for f in self.factors]            

        elif isinstance(other,DifferentialForm):
            ret.forms_list = [fl+[other] for fl in self.forms_list]
            ret.factors = self.factors

            ret.__remove_squares()
            ret.__remove_above_top()
            ret.__sort_form_sums()
            ret.__collect_forms()
        elif isinstance(other,DifferentialFormMul):
            for i in range(len(self.forms_list)):
                for j in range(len(other.forms_list)):
                    ret.forms_list.append(self.forms_list[i]+other.forms_list[j])
                    ret.factors.append(self.factors[i]*other.factors[j])

            ret.__remove_squares()
            ret.__remove_above_top()
            ret.__sort_form_sums()
            ret.__collect_forms()
        else:
            raise NotImplementedError
        
        return ret

    def __radd__(self,other): return self + other
    def __neg__(self):
        ret = DifferentialFormMul()
        ret.forms_list = self.forms_list
        ret.factors = [-f for f in self.factors]
        return ret
    
    def __sub__(self,other): return self + (-other)
    def __rsub__(self,other): return other + (-self)

    def __remove_squares(self):
        i = 0
        while i < len(self.forms_list):
            deled = False
            for j in range(len(self.forms_list[i])):
                if self.forms_list[i][j].degree %2 == 1:
                    if len([k for k,e in enumerate(self.forms_list[i]) if e == self.forms_list[i][j]]) > 1:
                        del self.forms_list[i]
                        del self.factors[i]
                        deled = True
                        break
            if not deled: i+=1
        
    def __remove_above_top(self):
        i = 0
        while i < len(self.forms_list):
            if sum([f.max_degree for f in self.forms_list[i]]) > self.max_degree:
                del self.forms_list[i]
                del self.factors[i]
                continue
            i += 1

    def __sort_form_sums(self):
        for i in range(len(self.forms_list)):
            bubble_factor = 1
            for j in range(len(self.forms_list[i])):
                for k in range(j,len(self.forms_list[i])):
                    if self.forms_list[i][j] < self.forms_list[i][k]:
                        temp = self.forms_list[i][j]
                        self.forms_list[i][j] = self.forms_list[i][k]
                        self.forms_list[i][k] = temp
                        bubble_factor *= (-1)**(self.forms_list[i][j].degree*self.forms_list[i][k].degree)
            self.factors[i] = self.factors[i]*bubble_factor
    
    def __collect_forms(self):
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
        
        self.forms_list = new_forms_list
        self.factors = new_factors
            
    def _repr_latex_(self):
        latex_str = "$" + "+".join([ "(" + latex(self.factors[i]) + ")" + r"\wedge".join([str(f) for f in self.forms_list[i]]) for i in range(len(self.forms_list))]) + "$"
        if latex_str == "$$":
            return "$0$"
        return latex_str
    

        
    