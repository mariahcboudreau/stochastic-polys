import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Colors from your scheme
GREEN="#519E3E"
ORANGE="#EF8636"
TEAL="#37B1AB"
SALMON="#F77B61"
BLUE="#3B75AF"
GRAY="#CCCCCC"

# Style parameters
SPINE_COLOR = '#333333'
FONT_SIZE_TITLE = 20
TITLE_PAD = 20

# Create custom branching process
def create_random_tree(max_levels=3, p_children=[0.2, 0.5, 0.3]):
    """Create a random tree where each node has a random number of children
    p_children: probability distribution for number of children (0, 1, or 2 children)
    """
    G = nx.DiGraph()
    G.add_node(0)
    current_level = [0]
    next_node = 1
    
    for level in range(max_levels):
        next_level = []
        for parent in current_level:
            # Random number of children (0, 1, or 2) based on p_children
            n_children = np.random.choice(len(p_children), p=p_children)
            for _ in range(n_children):
                G.add_node(next_node)
                G.add_edge(parent, next_node)
                next_level.append(next_node)
                next_node += 1
        if not next_level:  # Stop if no nodes in next level
            break
        current_level = next_level
    return G

def create_percolation_graph(n_nodes=10, edge_probability=0.3, infection_probability=0.2):
    """Create a random graph and mark some edges as infected"""
    # Create Erdos-Renyi random graph
    G = nx.erdos_renyi_graph(n_nodes, edge_probability, directed=False)
    
    # Randomly mark edges as infected
    edge_status = {}
    for edge in G.edges():
        edge_status[edge] = np.random.random() < infection_probability
        edge_status[(edge[1], edge[0])] = edge_status[edge]  # Add reverse edge (undirected)
    
    return G, edge_status

# Create figure
fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

# Create percolation graph
G, edge_status = create_percolation_graph(n_nodes=6, edge_probability=0.9, infection_probability=0.2)

# Create a subgraph with only infected edges
infected_edges = [(u, v) for (u, v) in G.edges() if edge_status[(u, v)]]
G_infected = G.edge_subgraph(infected_edges)

# Find nodes in connected components
nodes_in_components = set()
for component in nx.connected_components(G_infected):
    if len(component) > 1:  # Only consider components with more than one node
        nodes_in_components.update(component)

# Use spring layout for node positioning
pos = nx.spring_layout(G, k=1, seed=42)

# Draw non-infected edges (gray)
nx.draw_networkx_edges(G, pos, ax=ax,
                      edgelist=[e for e in G.edges() if not edge_status[e]],
                      edge_color=GRAY,
                      width=10,
                      alpha=0.5)

# # Draw infected edges (salmon)
# nx.draw_networkx_edges(G, pos, ax=ax,
#                       edgelist=[e for e in G.edges() if edge_status[e]],
#                       edge_color=SALMON,
#                       width=10,
#                       alpha=0.8)

# Draw nodes in components (green)
nx.draw_networkx_nodes(G, pos, ax=ax,
                      nodelist=list(nodes_in_components),
                      node_color=GREEN,
                      node_size=8000,
                      alpha=1)

# Draw other nodes (teal)
nx.draw_networkx_nodes(G, pos, ax=ax,
                      nodelist=list(set(G.nodes()) - nodes_in_components),
                      node_color=TEAL,
                      node_size=8000,
                      alpha=1)

# Clean up axes
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.title('Network Percolation', pad=20)
plt.tight_layout()
#plt.show()
plt.savefig("figures/percolation_process.svg", bbox_inches='tight')
plt.close()
