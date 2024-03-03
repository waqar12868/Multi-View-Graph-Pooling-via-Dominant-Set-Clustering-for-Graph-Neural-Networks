import torch
import argparse
import argparse
import numpy as np
import networkx as nx
from torch import Tensor
from numpy.linalg import norm
import torch.nn.functional as F
parser = argparse.ArgumentParser()
from itertools   import combinations
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from scipy.spatial.distance import cdist
from torch_geometric.utils import to_networkx
from sklearn.metrics.pairwise import cosine_similarity

parser.add_argument('--seed', type=int, default=123,
                    help='seed')
args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:3'

# Define the Dominant Pooling layer
class DominantClustering(torch.nn.Module):
    __all_ops__ = {
      'avg': lambda x: torch.mean(x, dim=0),
      'max': lambda x: torch.max(x, dim=0),
      'sum': lambda x: torch.sum(x, dim=0)
    }
    def __init__(self,agg_type: str):
        super(DominantClustering, self).__init__()
        assert agg_type in DominantClustering.__all_ops__, \
        "agg_type should be either 'max', 'sum' or 'avg'."
        self.agg_type  = agg_type
        self.operation = DominantClustering.__all_ops__[agg_type]

    def get_nodes_num(edge_index):
        # Check if edge_index is empty
        if edge_index.numel() == 0:
            # Handle the case of an empty edge_index (e.g., single node)
            return 1
        elif isinstance(edge_index, Tensor):
            # Find the number of nodes for a non-empty tensor
            nodes_num = int(edge_index.max()) + 1
        else:
            # Handle other cases (e.g., edge_index as a list or numpy array)
            nodes_num = max(edge_index.size(0), edge_index.size(1))
        return nodes_num
  
    def get_edges(edge_index):
        if isinstance(edge_index, SparseTensor):
            col, row, _ = edge_index.coo()
            edges = list(zip(row, col))
        elif isinstance(edge_index, Tensor):
            row, col = edge_index
            edges = list(zip(row.tolist(), col.tolist()))
        else:
            edges = edge_index.tolist()
        return edges
    
    def get_neighbours(edge_index, v):
        edges = DominantClustering.get_edges(edge_index)
        return set(
        map(
            lambda p: p[1],
            filter(lambda p: p[0] == v, edges)
        )
        )
    
    def to_networkx(edge_index, nodes_num):
        G = nx.Graph()
        G.add_nodes_from(range(nodes_num))
        edges = DominantClustering.get_edges(edge_index)
        for (u,v) in edges:
            G.add_edge(u, v)
        return G    
    
    #calculate the edge weights from node features using cosine similarity
    def compute_edge_weights(edge_index, node_features):
        edge_weights = []
        node_features = node_features.cpu().detach()   
        for i in range(edge_index.size(1)):
            source, target = edge_index[:, i].cpu()  
            weight = cosine_similarity(
                node_features[source].reshape(1, -1),
                node_features[target].reshape(1, -1)
            )[0, 0]
            edge_weights.append(weight)
        return torch.tensor(edge_weights) 

    def compute_edge_weights_from_node_features(edge_index, node_features):
      G = nx.Graph()
      edge_weights = DominantClustering.compute_edge_weights(edge_index, node_features)
      for i, (u, v) in enumerate(edge_index.t().tolist()):
          G.add_edge(u, v, weight=edge_weights[i].item())
      return G
    
    #calculate the edge weights from graph structural information using node centrality
    def compute_edge_weights_from_node_centrality(G):
        centrality = nx.degree_centrality(G)
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = centrality[u] + centrality[v]
        return G
    
    #calculate the edge weights from edge features using cosine similarity
    def compute_edge_weights_using_cosine_similarity(edge_features):
        edge_weights = []
        edge_features = edge_features.cpu().detach()   # Move node_features to CPU
        # Normalize edge features to unit vectors
        edge_features_normalized = F.normalize(edge_features, p=2, dim=1)
        # Compute cosine similarity between all pairs of edges
        similarity_matrix = cosine_similarity(edge_features_normalized)
        edge_weights = torch.tensor(np.mean(similarity_matrix, axis=1)).to(args.device)
        return edge_weights
    
    def edge_index_to_weighted_graph_with_edge_features(edge_index, edge_weights):
        G = nx.Graph()
        for i, (u, v) in enumerate(edge_index.t().tolist()):
            # Use edge_weights directly
            G.add_edge(u, v, weight=edge_weights[i].item())
        return G
    
    def rbf_kernel(G):
        # Convert NetworkX graph to a SciPy sparse matrix
        A = nx.to_numpy_array(G)
        # Use sparse matrix exponentiation
        K = np.exp(A)
        np.fill_diagonal(K, 0)
        return K
    
    def rbf_kernel1_similarity_matrix(G, sigma=1.0):
        # Convert edge_index to adjacency matrix
        adj_matrix = nx.to_numpy_array(G)
        # Compute pairwise distances
        pairwise_dists = cdist(adj_matrix, adj_matrix, 'euclidean')
        # Compute the RBF kernel
        K = np.exp(-pairwise_dists ** 2 / (2 * sigma ** 2))
        np.fill_diagonal(K, 0)
        return K

    def RD(A, x=None, epsilon=2.0e-4):
        """A is a similarity matrix and compute the dominant sets for the A with the
            replicator dynamics optimization approach. 
            Convergence is reached when x changes less than epsilon.
            """
        if x is None:
            x = np.ones(A.shape[0])/float (A.shape[0])
        
        distance = epsilon*2.0
        
        while distance > epsilon:
            x_old = x.copy()
            x = x* A.dot(x)
            x = x/ x.sum()
            distance = norm(x-x_old)
        return x
    
    def Peeloff(G):
        S= DominantClustering.rbf_kernel(G)
        nodes = np.arange(S.shape[0])  # Initialize nodes based on the shape of S
        beta_idx = []
        dominant_nodes=[]
        #clique_scores =[]
        while S.shape[0] > 1:
            x = DominantClustering.RD(S)
            beta_threshold = 1.0e-6 #filter the values
            beta_idx = np.where(x>=beta_threshold)[0]#.tolist() #extract the indices 
            c = nodes[beta_idx]
            dominant_nodes.append(c)
            remaining_indices = np.where(x <= beta_threshold)[0]
            S = S[remaining_indices][:, remaining_indices]
            nodes = nodes[remaining_indices]
        return dominant_nodes
    
    def clusters_edges(edge_index, considered_cliques, clusters, nodes_num):
        def map_and_unite(func, elems):
            if len(elems) == 0:
                return set()
            return set.union(
                *list(
                map(func, elems)
                )
            )
        neighbourhood = {
        n: DominantClustering.get_neighbours(edge_index, n) for n in range(nodes_num)
        }
        edges = set()
        # Add `sibling` edges:
        for _, values in clusters.items():
            if len(values) > 1:
                for pair in set(combinations(values, 2)):
                    edges.add(pair)
        # Compute the clusters' `siblings`
        siblings = [
        map_and_unite(
            lambda c: set(clusters[c]),
            cluster
        )
        for cluster in considered_cliques
        ]
        # Add `intra cluster` edges:
        for idx, clique in enumerate(considered_cliques):
            # Compute union of neighbors in the cluster:
            clique_neighbors = \
                map_and_unite(lambda v: neighbourhood[v], clique)
            # Map clique_neighbors's nodes to new nodes
            # and substruct siblings
            dual_neighbors = map_and_unite(
                lambda x: set(clusters[x]),
                clique_neighbors
            ).difference(siblings[idx])
            # Add `intra cluster` edges:
            for v in dual_neighbors:
                edges.add(tuple(sorted([idx, v])))
        return list(edges)
    
    def forward(self, x, edge_index,edge_attr=None, weight_calculation_method='node_features'):
        # Calculate edge weights based on the selected method
        if weight_calculation_method == 'node_features':
            G = DominantClustering.compute_edge_weights_from_node_features(edge_index,x)
        elif weight_calculation_method == 'edge_features' and edge_attr is not None:
            edge_weights = DominantClustering.compute_edge_weights_using_cosine_similarity(edge_attr)
            G = DominantClustering.edge_index_to_weighted_graph_with_edge_features(edge_index, edge_weights)
        elif weight_calculation_method == 'node_centrality':
            g = Data(x=x, edge_index=edge_index)
            g = to_networkx(g, to_undirected=True)
            G = DominantClustering.compute_edge_weights_from_node_centrality(g)
        else:
            raise ValueError("Unsupported weight calculation method.")
        
        dominant_clusters= DominantClustering.Peeloff(G)
        Hcoarse = torch.stack(
        [
            self.operation(torch.index_select(x,0,torch.tensor(clique, dtype=torch.long).to(args.device)
            )) for clique in dominant_clusters
        ]
        )
        # Step 1: Calculate cluster connectivity
        connectivity_counts = torch.zeros((len(dominant_clusters), len(dominant_clusters)), device=x.device)
        for i, cluster_i in enumerate(dominant_clusters):
            for j, cluster_j in enumerate(dominant_clusters):
                if i != j:  # Ignore self-connectivity
                    connectivity_counts[i, j] = sum(
                        1 for u, v in zip(edge_index[0], edge_index[1])
                        if (u.item() in cluster_i and v.item() in cluster_j) or (u.item() in cluster_j and v.item() in cluster_i)
                    )
        # Step 2: Aggregate connectivity information
        # Here, we simply sum the connectivity counts for each cluster
        aggregated_connectivity = connectivity_counts.sum(dim=1)
        return Hcoarse, dominant_clusters