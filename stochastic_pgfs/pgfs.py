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

    def normalize(self):
        """
        Normalizes coefficients
        """
        self.coef / np.sum(self.coef)


# #generate PGF corresponding to self consistent equation G(u) = u
# def make_G_u_minus_u(G):
#     G_prime = G.derivative()
#     G_1 = PGF(G_prime.coef / G_prime(1.0))
#     G_1_minus_u_coef = np.copy(G_1.coef)
#     G_1_minus_u_coef[1] -= 1
#     return G_1_minus_u_coef


# generate PGF corresponding to self consistent equation G(u) = u without taking derivative inherantly
def make_G_u_minus_u(coefs):
    G = PGF(coefs)
    G.normalize()
    G_minus_u_coef = np.copy(G.coef)
    G_minus_u_coef[1] -= 1
    return G_minus_u_coef

#numerical inversion of the PGF
def numerical_inversion(pgf,N = 100):
    #G = np.vectorize(pgf)
    G = pgf
    n = np.arange(N)
    c = np.exp(2*np.pi*1j*n/N)
    pn = abs(np.fft.fft(G(c))/N)
    return pn

#percolated pgf
def percolated_pgf(my_pgf,x,T):
    new_x = 1-T+T*x
    return my_pgf(new_x)





