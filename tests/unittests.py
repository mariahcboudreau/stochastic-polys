
#test poly coefs 
import numpy as np 
from stochastic_pgfs.laub_xia_algo import polynomial_roots,companion_matrix, is_real, in_bounds,_solve_self_consistent_equation
from stochastic_pgfs.pgfs import PGF, make_G_u_minus_u
from scipy.stats import binom,poisson
import matplotlib.pyplot as plt





def test_companion_matrix():
    poly_coefs = [1,-2,-1,2]#X^3-2X^2-X+2
    C = companion_matrix(poly_coefs)
    correct_companion_matrix = np.array([[ 0.,  0., -2.],[ 1.,  0.,  1.],[ 0.,  1.,  2.]])
    assert np.allclose(C,correct_companion_matrix)

def test_polynomial_roots():
    poly_coefs = [1,-2,-1,2]#X^3-2X^2-X+2
    correct_roots = np.array([2., 1., -1.])
    poly_roots = polynomial_roots(poly_coefs)
    assert np.allclose(np.sort(poly_roots),np.sort(correct_roots))

def test_random_poly(): 
    #check if polyonmial with random roots is equal to numpy root solver
    np.random.seed(0)
    poly_coefs = np.random.randn(5)
    poly_roots = polynomial_roots(poly_coefs)
    assert np.allclose(np.polyval(poly_coefs,poly_roots),0)

    


def test_giant_component_size(): 
    s_list = []
    n = 15
    lmbd_list = np.linspace(0.99,1.01)
    for lmbd in lmbd_list:
        degree_sequence = np.zeros(n+1)
        degree_sequence = poisson.pmf(np.arange(0,n+1),lmbd)
        roots = _solve_self_consistent_equation(degree_sequence)
        s = 1-np.min(roots)
        #s = 1-np.max(roots)
        s_list.append(s)
   
    #test the the giant component is 0 below the critical point and 1 above the critical point
    subcrictial = np.isclose(s_list[0],0) #subcritcial is 0 
    supercritical = s_list[-1] > 0  #supercritical is 1
    assert np.logical_and(subcrictial,supercritical)





            
            
 

















