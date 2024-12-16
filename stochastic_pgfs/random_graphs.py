import numpy as np
import scipy.stats as stats
import networkx as nx
#define degree sequence for ER graph 

def er_degree_sequence(p,n,support = np.arange(0,100)):
    #poisosn distribution
    degree_sequence = stats.poisson.pmf(support,n*p)
    return degree_sequence


def power_law(n,alpha):
    degree_sequence = n**-alpha
    return degree_sequence/sum(degree_sequence)


def powerlaw_degree_sequence(alpha,N_max = 100):
    degree_sequence = power_law(np.arange(1, N_max, 1), alpha)
    return np.insert(degree_sequence, 0, 0)



def mean_power_law(minval, maxval, alpha):
    """
    Calculates the mean of a power law distribution given the minimum value, maximum value, and alpha.

    Parameters:
    ----------
    minval : float
        The minimum value of the distribution.
    maxval : float
        The maximum value of the distribution.
    alpha : float
        The exponent of the power law distribution.

    Returns:
    -------
    float
        The mean of the power law distribution.
    """
    alpha= -alpha
    
    if alpha == -1:
        num = maxval - minval
        den = np.log(maxval) - np.log(minval)
        return num / den
    elif alpha == -2:
        num = np.log(maxval) - np.log(minval)
        den = 1 / minval - 1 / maxval
        return num / den
    else:
        num = (minval ** (2 + alpha) - maxval ** (2 + alpha)) / (-alpha - 2)
        den = (minval ** (1 + alpha) - maxval ** (1 + alpha)) / (-alpha - 1)
        return num / den
    
    

def power_law_variance(alpha, k_min, k_max):
    """
    Calculate the variance of the degree distribution for a power-law graph.

    Parameters:
    alpha : float
        The power-law exponent (must be greater than 1).
    k_min : float
        The minimum degree cutoff (must be positive).
    k_max : float
        The maximum degree cutoff (must be greater than k_min).

    Returns:
    variance : float
        The variance of the degree distribution, or 'inf' if it diverges.

    Raises:
    ValueError: If input parameters are invalid.
    ZeroDivisionError: If a division by zero occurs in calculations.
    """
    # Input validation
    if k_min <= 0 or k_max <= 0:
        raise ValueError("k_min and k_max must be positive numbers.")
    if k_min >= k_max:
        raise ValueError("k_min must be less than k_max.")
    if alpha <= 1:
        raise ValueError("alpha must be greater than 1 to avoid division by zero.")

    # Calculate the normalization constant C
    denominator = k_min**(1 - alpha) - k_max**(1 - alpha)
    if denominator == 0:
        raise ZeroDivisionError("Denominator in normalization constant C is zero.")
    C = (alpha - 1) / denominator

    # Calculate the first moment (average degree)
    if alpha == 2:
        # Handle the special case when (2 - alpha) == 0
        avg_degree = C * np.log(k_max / k_min)
    else:
        numerator_avg_degree = k_max**(2 - alpha) - k_min**(2 - alpha)
        avg_degree = C * numerator_avg_degree / (2 - alpha)

    # Calculate the second moment
    if alpha == 3:
        # Handle the special case when (3 - alpha) == 0
        second_moment = C * np.log(k_max / k_min)
    else:
        numerator_second_moment = k_max**(3 - alpha) - k_min**(3 - alpha)
        second_moment = C * numerator_second_moment / (3 - alpha)

    # Variance is the second moment minus the square of the average degree
    variance = second_moment - avg_degree**2

    # Check for negative variance due to numerical errors
    if variance < 0:
        variance = 0.0

    # Check for infinite variance when alpha <= 2
    if alpha <= 2:
        variance = np.inf

    return variance
    
    

    
def poisson_degree_sequence(lambd,N_max = 100):
    #poisson distribution
    degree_sequence = stats.poisson.pmf(np.arange(0,N_max),lambd)
    return degree_sequence

def make_config_model(degree_sequence_func,num_nodes = 200, **kwargs):
    degree_sequence = degree_sequence_func(**kwargs)
    deg_seq = np.round(degree_sequence*num_nodes)
    
    if sum(deg_seq) % 2 != 0:
        deg_seq[1] += 1
    
    G = nx.configuration_model(deg_seq.astype(int))
    return G

def create_er_graph(p, n):
    G = nx.erdos_renyi_graph(n, p)
    return G


def create_cm_powerlaw(alpha, N_max):
    G = make_config_model(powerlaw_degree_sequence, alpha=alpha, N_max=N_max)
    return G

def get_largest_connected_component(G):
    # Get the largest connected component
    lcc = max(nx.connected_components(G), key=len)
    # Create a subgraph from the LCC
    return G.subgraph(lcc).copy()


def plot_graph(G,node_options,edge_options,ax):
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    G = get_largest_connected_component(G)
    pos = nx.spring_layout(G,
                           k = 1e-2,
                           seed = 1)
    
    nx.draw_networkx_nodes(G,
            **node_options,
            ax = ax,
            pos = pos
            )
    
    nx.draw_networkx_edges(G,
            **edge_options,
            ax = ax,
            pos = pos
            )
    ax.set_axis_off()
