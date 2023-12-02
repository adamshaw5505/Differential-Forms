from sympy import Symbol, I

class DifferentialForm:
    def __init__(self,symbol,degree=0,max_degree=4, exact=False):
        assert degree > 0
        assert degree <= max_degree
        self.max_degree = degree
        self.degree = degree
        self.symbol = symbol

    def __mul__(self,other):
        if isinstance(other,Symbol):
            return DifferentialFormMul(self,other)
        elif isinstance(other,int):
            return DifferentialFormMul(self,I*other/I)


    def _repr_latex_(self):
        return self.symbol._repr_latex_()


class DifferentialFormMul():
    def __init__(self,form:DifferentialForm,factor:Symbol):
        self.forms_list = [[form]]
        self.factors = [factor]

    def _repr_latex_(self):
        latex_str = "$" + "+".join(["(" + self.factors[i]._repr_latex_() + ")" + r'\wedge'.join(self.forms_list[i]) for i in range(len(self.forms_list))]) + "$"
        if latex_str == "$$":
            latex_str = "0"
        
        return latex_str
    

        
    