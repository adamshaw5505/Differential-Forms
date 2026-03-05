# TODO: Give these operators on differential forms

from . import *

def J1(thetas,twoforms,simplify=False):
    """Computes the J1 operator on R3 valued 1-forms in 4D given a triple of 2-forms"""
    g_UU = twoforms[0].manifold.get_inverse_metric()
    S_iDU = [Contract(s.to_tensor()*g_UU,(1,2)).simplify() for s in twoforms]
    if simplify:
        return [sum([LeviCivita(i,j,k)*Contract(S_iDU[j]*thetas[k].to_tensor(),(1,2)) for j,k in drange(3,2)]).to_differentialform().simplify() for i in range(3)]
    return [sum([LeviCivita(i,j,k)*Contract(S_iDU[j]*thetas[k].to_tensor(),(1,2)) for j,k in drange(3,2)]).to_differentialform() for i in range(3)]

def J2(Bi,twoforms):
    """Special product that exists in the context of SU(2) structures. """
    Bi_DD = [2*b.to_tensor() for b in Bi]
    g_UU = twoforms[0].manifold.get_inverse_metric()
    Si_DU = [Contract(s.to_tensor()*g_UU,(1,2)) for s in twoforms]

    return [sum([LeviCivita(i,j,k)*Contract(Si_DU[j]*Bi_DD[k],(1,2)) for j,k in drange(3,2)]).to_differentialform()/Number(2) for i in range(3)]


def ExteriorSO3GaugeDerivative(thetas,A_i,manifold=None):
    """ SO(3) Gauge Covariant Derivative """
    result = [d(thetas[i],manifold) + sum([LeviCivita(i,j,k)*A_i[j]*thetas[k] for j,k in drange(3,2)]) for i in range(3)]
    return result

def ExteriorSO3GaugeCoDerivative(thetas,A_i,manifold=None):
    """ SO(3) Covaraint Exterior CoDerivative """
    hodge_theta = [Hodge(t,manifold) for t in thetas]
    dA_hodge_theta = dA(hodge_theta,A_i,manifold)
    return [Hodge(t,manifold) for t in dA_hodge_theta]

def GetSelfDualTwoForm(frame,orientation=1,signature=1):
    """ Return Self-Dual 2-forms given a frame """
    assert(len(frame)==4)

    sigma = 1 if signature == 1 else I
    return [frame[0]*frame[i+1]*sigma-sum([int(LeviCivita(i,j,k))*frame[j+1]*frame[k+1] for j,k in drange(3,2)])*orientation/Number(2) for i in range(3)]

def GetSelfDualConnections(twoforms,signature=1,orientation=1):
    """ Get the self-dual connections from the self-dual 2-forms """
    star_dS_i = [Hodge(d(si)) for si in twoforms]
    J1_star_dS_i = J1(star_dS_i,twoforms)
    sigma = Number(1) if signature == 1 else I
    return [orientation*sigma/Number(2)*(J1_star_dS_i[i] - orientation*star_dS_i[i]) for i in range(3)]

def GetSO3Curvature(connections):
    """ Computes the SO(3,C) curvature of an SO(3,C) connection """
    return [connections[i] + Number(1,2)*sum([LeviCivita(i,j,k)*connections[j]*connections[k] for j,k in drange(3,2)]) for i in range(3)]

def GetUrbantkeMetric(twoforms):
    """ Computes metric from a triple of 2-forms """
    assert(twoforms[0].manifold.dimension == 4)
    TraceSS = simplify(sum([s*s for s in twoforms]).factors[0])
    if TraceSS == 0: return Number(0)
    manifold = twoforms[0].manifold
    vects = manifold.vectors
    basis = manifold.basis

    f = 1 if manifold.signature == 1 else I
    g_DD = 0
    for K,J in drange(4,2):
        fact = sum([LeviCivita(i,j,k)*twoforms[i].insert(vects[K])*twoforms[j].insert(vects[J])*twoforms[k] for i,j,k in drange(3,3)])
        if fact == 0: continue
        g_DD += fact.factors[0]*basis[K].to_tensor()*basis[J].to_tensor()
    return -g_DD/TraceSS