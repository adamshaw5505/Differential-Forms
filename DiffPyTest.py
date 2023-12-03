from sympy import Symbol

def bubble_sort_forms(forms,degrees):
    bubbled_factor = 1
    n = len(forms)
    swapped = False
    for i in range(n-1):
        for j in range(0,n-i-1):
            if str(forms[j]) > str(forms[j+1]):
                forms[j+1], forms[j] = forms[j], forms[j+1]
                degrees[j+1], degrees[j] = degrees[j], degrees[j+1]
                swapped = True

                bubbled_factor *= (-1)**(degrees[j+1]*degrees[j])
        if not swapped:
            break
    return forms, degrees, bubbled_factor

def bubble_tester(forms,degrees):
    return bubble_sort_forms(forms,degrees)

class DifferentialForm:
    def __init__(self, symbol, degree=0, max_degree=4, basis=None):
        assert degree >= 0
        assert degree <= max_degree
        # assert isinstance(symbol,Symbol)
        self.max_degree = max_degree
        if symbol != None:
            self.forms_list = [[symbol]]
            self.degree_list = [[1]]
            self.factors_list = [1]
        else:
            self.forms_list = []
            self.degree_list = []
            self.factors_list = []
    
    def __reduce_form(self):
        """ Think of a good way to simplify a form, that is:
                [X] Elminate a term if the degree is too high
                [X] Kill a term if there are two of the same term
                [X] Collect terms together if they are the same
                [X] Kill remaining terms that have factors of zero.
                [X] Order factors and get permutation to find sign (hardest problem)

                [X] Fix order of operations on the forms

            
            Algorithm is implmenented but probably slow/buggy
            Maybe implement when the forms are added/multiplied together so that it is more optimised?
        """
        i = 0
        while i < len(self.forms_list):
            # Check if the degree of the product of forms is above the "max_degree" or if the product of form contains multiple of the same form.
            if sum(self.degree_list[i]) > self.max_degree or len(set(self.forms_list[i])) < len(self.forms_list[i]):
                del self.degree_list[i]
                del self.forms_list[i]
                del self.factors_list[i]
                continue
            i+=1
        
        new_forms_list = []
        new_degree_list = []
        new_factors_list = []
        # Loop through the forms to find the unique forms list and collect the factors of all unique forms.
        for i in range(len(self.forms_list)):
            if self.forms_list[i] not in new_forms_list:
                new_forms_list.append(self.forms_list[i])
                new_degree_list.append(self.degree_list[i])
                new_factors_list.append(self.factors_list[i])
            else:
                idx = new_forms_list.index(self.forms_list[i])
                new_factors_list[idx] += self.factors_list[i]

        #Bubble sort each term and keep track of the factor
        for i in range(len(new_forms_list)):
            bubbled_forms, bubbled_degrees, bubbled_factor = bubble_sort_forms(new_forms_list[i],new_degree_list[i])
            new_forms_list[i] = bubbled_forms
            new_factors_list[i] = new_factors_list[i]*bubbled_factor
            
        self.forms_list = new_forms_list
        self.degree_list = new_degree_list
        self.factors_list = new_factors_list

        new_forms_list = []
        new_degree_list = []
        new_factors_list = []

        # Loop through the forms to find the unique forms list and collect the factors of all unique forms.
        for i in range(len(self.forms_list)):
            if self.forms_list[i] not in new_forms_list:
                new_forms_list.append(self.forms_list[i])
                new_degree_list.append(self.degree_list[i])
                new_factors_list.append(self.factors_list[i])
            else:
                idx = new_forms_list.index(self.forms_list[i])
                new_factors_list[idx] += self.factors_list[i]

        # Finally check which factors are zero and remove them
        while i < len(new_forms_list):
            if new_factors_list[i] == 0:
                del new_degree_list[i]
                del new_forms_list[i]
                del new_factors_list[i]
                continue
            i+=1


        self.forms_list = new_forms_list
        self.degree_list = new_degree_list
        self.factors_list = new_factors_list

    def __rmul__(self, other):
        new_form = DifferentialForm(None)
        if isinstance(other,DifferentialForm):
            for i in range(len(self.forms_list)):
                for j in range(len(other.forms_list)):
                    new_form.forms_list.append(other.forms_list[i]+self.forms_list[j])
                    new_form.degree_list.append(other.degree_list[i]+self.degree_list[j])
                    new_form.factors_list.append(other.factors_list[i]*self.factors_list[j])
        elif isinstance(other,int):
            new_form.forms_list = self.forms_list
            new_form.degree_list = self.degree_list
            new_form.factors_list = [other*x for x in self.forms_list]

        return new_form
    
    def __mul__(self, other):
        new_form = DifferentialForm(None)
        if isinstance(other,DifferentialForm):
            for i in range(len(self.forms_list)):
                for j in range(len(other.forms_list)):
                    new_form.forms_list.append(self.forms_list[i]+other.forms_list[j])
                    new_form.degree_list.append(self.degree_list[i]+other.degree_list[j])
                    new_form.factors_list.append(self.factors_list[i]*other.factors_list[j])
        elif isinstance(other,int):
            new_form.forms_list = self.forms_list
            new_form.degree_list = self.degree_list
            new_form.factors_list = [other*x for x in self.forms_list]

        return new_form

    def __sub__(self,other):
        new_form = DifferentialForm(None)
        if isinstance(other,DifferentialForm):
            new_form.forms_list = self.forms_list + other.forms_list
            new_form.degree_list = self.degree_list + other.degree_list
            new_form.factors_list = self.factors_list + [-x for x in other.factors_list]
        new_form.__reduce_form()
        return new_form

    
    def __add__(self, other):
        new_form = DifferentialForm(None)
        if isinstance(other,DifferentialForm):
            new_form.forms_list = self.forms_list + other.forms_list
            new_form.degree_list = self.degree_list + other.degree_list
            new_form.factors_list = self.factors_list + other.factors_list
        
        new_form.__reduce_form()
        return new_form

    def subs(self,other):
        if isinstance(other,DifferentialForm):
            
        

    def __str__(self):
        # return '+'.join(['^'.join([str(self.forms_list[i][j]) for j in range(len(self.forms_list[i]))]) for i in range(len(self.forms_list))])
        

    def _repr_latex_(self):
        latex_str = "$" + '+'.join([ "(" + str(self.factors_list[i]) + ")" + r' \wedge '.join([str(self.forms_list[i][j]) for j in range(len(self.forms_list[i]))]) for i in range(len(self.forms_list))]) + "$"
        if latex_str == "$$":
            latex_str = "$0$"
        return latex_str
        
