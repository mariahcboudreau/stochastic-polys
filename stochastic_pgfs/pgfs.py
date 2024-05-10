import numpy as np 
##class defines probability generating fucntion
class PGF:
    def __init__(self, coef):
        """
        coef: list of coefficients of the PGF, starting with the constant term
        """
        self.coef = np.array(coef, dtype=float)
    
    def __call__(self, x):
        """
        x: point at which to evaluate the PGF
        """
        return sum(self.coef[i] * x**i for i in range(len(self.coef)))
    
    def derivative(self):
        """
        Returns the derivative of the PGF
        """
        deriv_coefs = self.coef[1:] * np.arange(1, len(self.coef))
        return PGF(deriv_coefs)

#generate PGF corresponding to self consistent equation G(u) = u
def make_G_u_minus_u(G):
    G_prime = G.derivative()
    G_1 = PGF(G_prime.coef / G_prime(1.0))
    G_1_minus_u_coef = np.copy(G_1.coef)
    G_1_minus_u_coef[1] -= 1
    return G_1_minus_u_coef
