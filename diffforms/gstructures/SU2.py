# TODO: Give J1 and J2 better names

from diffforms import *

def J1(thetas : list[DifferentialFormMul], su2_structures : list[DifferentialFormMul], simplify : bool = False) -> list[DifferentialFormMul]:
    """ Operator on SU(2)-valued 1-forms in 4 dimensions"""
    g_UU = su2_structures[0].manifold.get_inverse_metric()
    S_iDU = [Contract(s.to_tensor()*g_UU,(1,2)).simplify() for s in su2_structures]
    if simplify:
        return [sum([LeviCivita(i,j,k)*Contract(S_iDU[j]*thetas[k].to_tensor(),(1,2)) for j,k in drange(3,2)]).to_differentialform().simplify() for i in range(3)]
    return [sum([LeviCivita(i,j,k)*Contract(S_iDU[j]*thetas[k].to_tensor(),(1,2)) for j,k in drange(3,2)]).to_differentialform() for i in range(3)]

def J2(Bi : list[DifferentialFormMul], su2_structures : list[DifferentialFormMul]) -> list[DifferentialFormMul]:
    """ Operator on SU(2)-valued 2-forms in 4-dimensions """
    Bi_DD = [2*b.to_tensor() for b in Bi]
    g_UU = su2_structures[0].manifold.get_inverse_metric()
    Si_DU = [Contract(s.to_tensor()*g_UU,(1,2)) for s in su2_structures]
    return [sum([LeviCivita(i,j,k)*Contract(Si_DU[j]*Bi_DD[k],(1,2)) for j,k in drange(3,2)]).to_differentialform()/Number(2) for i in range(3)]


def ExteriorSU2GaugeDerivative(thetas : list[DifferentialFormMul], A_i : list[DifferentialFormMul], manifold : Manifold = None) -> list[DifferentialFormMul]:
    """ SU(2) Gauge Covariant Derivative """
    result = [ExteriorDerivative(thetas[i],manifold) + sum([LeviCivita(i,j,k)*A_i[j]*thetas[k] for j,k in drange(3,2)]) for i in range(3)]
    return result

def ExteriorSU2GaugeCoDerivative(thetas : list[DifferentialFormMul], A_i : list[DifferentialFormMul], manifold : Manifold = None) -> list[DifferentialFormMul]:
    """ SU(2) Covariant Exterior CoDerivative """
    hodge_theta = [Hodge(t,manifold) for t in thetas]
    dA_hodge_theta = ExteriorSO3GaugeDerivative(hodge_theta,A_i,manifold)
    return [Hodge(t,manifold) for t in dA_hodge_theta]

def GetSU2Structures(frame : list[DifferentialFormMul], orientation : int = 1, signature : list[int] = None) -> list[DifferentialFormMul]:
    """ Computes the self-dual 2-form SU(2) structures from a given frame """
    assert(len(frame)==4)
    if signature == None:
        signature = frame[0].manifold.signature

    sig_prod = prod(signature)
    eta = diag(*signature[1:])
    sigma = 1 if sig_prod == 1 else I
    return [sigma*frame[0]*frame[i+1]+Number(orientation*signature[0],2)*sum([eta[i,l]*LeviCivita(l,j,k)*frame[j+1]*frame[k+1] for l,j,k in drange(3,3)]) for i in range(3)]

def GetSU2Connections(su2_structures : list[DifferentialForm], signature : list[int] = None, orientation : int = 1) -> list[DifferentialFormMul]:
    """ Computes the connections (or torsion) for a given SU(2) structure """
    if signature == None:
        sign = su2_structures[0].manifold.signature_prod
    star_dS_i = [Hodge(ExteriorDerivative(si)) for si in su2_structures]
    J1_star_dS_i = J1(star_dS_i,su2_structures)
    sigma = Number(1) if sign == 1 else I
    return [orientation*sigma/Number(2)*(J1_star_dS_i[i] - orientation*star_dS_i[i]) for i in range(3)]

def GetSU2Curvature(connections : list[DifferentialFormMul]) -> list[DifferentialFormMul]:
    """ Computes the curvature of the SU(2) structures """
    return [ExteriorDerivative(connections[i]) + Number(1,2)*sum([LeviCivita(i,j,k)*connections[j]*connections[k] for j,k in drange(3,2)]) for i in range(3)]

def GetUrbantkeMetric(su2_structures : list[DifferentialFormMul]) -> Tensor:
    """ Computes metric from a triple of 2-forms """
    man = su2_structures[0].manifold
    assert(man.dimension == 4)
    TraceSS = simplify(sum([s*s for s in su2_structures]).factors[0])
    if TraceSS == 0: return Number(0)
    vects = man.vectors
    basis = man.basis

    f = 1 if man.signature_prod == 1 else I
    g_DD = 0
    for K,J in drange(4,2):
        fact = sum([LeviCivita(i,j,k)*su2_structures[i].insert(vects[K])*su2_structures[j].insert(vects[J])*su2_structures[k] for i,j,k in drange(3,3)])
        if fact == 0: continue
        g_DD += fact.factors[0]*basis[K].to_tensor()*basis[J].to_tensor()
    return -g_DD/TraceSS

def GetSU2LieAlgebraFromTwoForm(twoform : DifferentialFormMul, su2_structure : list[DifferentialFormMul]) -> list[Expr]:
        """Returns the SU(2) Lie algebra element for a 2-form, given a triple of SU(2) structures, in the vector representation

        Arguments:
            - twoform(List[DifferentialFormMul]):  Generic 2-form of which the self-dual matrix will be returned.
            - su2_structure(List[DifferentialFormMul]): Triple of self-dual 2-forms.
        
        Returns:
            - Matrix of 3x3 components (Can be complex).
        """
        
        assert(twoform.get_degree() == 2)
        assert([s.get_degree() for s in su2_structure] == [2, 2, 2])

        volSD = sum([s*s for s in su2_structure]).factors[0]/(1 if su2_structure[0].manifold.signature_prod == 1 else I)
        return [(twoform*s).factors[0]/(2*volSD) for s in su2_structure]