from stochastic_pgfs.laub_xia_algo import *

# # Test kappa_SCE with the polynomial coefficients for 2x^2 - 3x + 1
test_kappa_sce = l_x_algo([2, -3, 1], is_pgf = False, K = 10, conditions=[is_real, in_bounds])  # Reduced K for simplicity
print(test_kappa_sce) ### Check this out. 
