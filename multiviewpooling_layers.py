
import math
import torch
import argparse
import argparse
import torch.nn as nn
from numpy.linalg import norm
import torch.nn.functional as F
parser = argparse.ArgumentParser()
from scipy.spatial.distance import cdist
from torch.utils.data import random_split
from torch_geometric.nn import GraphConv, GATConv
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch_geometric.utils import to_networkx
from Dominant_Clustering import DominantClustering
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
parser.add_argument('--seed', type=int, default=123,
                    help='seed')
args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:3'
# Define the Clique Pooling layer
class DSMVPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, mlp_hidden=64):
        super(DSMVPool, self).__init__()
        self.pooling_ratio = ratio
        self.dominantset = DominantClustering('avg')
        self.inter_channel_gcn = InterChannelGCN(in_channels,in_channels)
        self.attention_GATConv = GATConv(in_channels, 1, heads=1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )
    def create_relationship_matrix(self,cliques, top_nodes):
      num_cliques = len(cliques)
      num_top_nodes = len(top_nodes)
      relationship_matrix = torch.zeros((num_top_nodes,num_cliques))
      # Fill the relationship matrix
      for i, topk_node in enumerate(top_nodes):
        for j, clique in enumerate(cliques):
            if topk_node in clique:
                relationship_matrix[i, j] = 1 
      return relationship_matrix
    
    def forward(self, x, edge_index, batch):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        Hcoarse, dominant_clusters = self.dominantset(x, edge_index, weight_calculation_method='node_centrality')
      
        feature_score = self.mlp(x).squeeze()
        feature_topk_indices = topk(feature_score, self.pooling_ratio, batch)
        feature_view3 = x[feature_topk_indices]

        feature_global_relationship_matrix = self.create_relationship_matrix(dominant_clusters, feature_topk_indices).to(args.device)
        fuse1 = self.inter_channel_gcn(feature_view3, Hcoarse, feature_global_relationship_matrix)

        localview_score = self.attention_GATConv(x,edge_index).squeeze()
        localview_topk_indices = topk(localview_score, self.pooling_ratio, batch)
        local_view1 = x[localview_topk_indices]
        
        local_global_relationship_matrix = self.create_relationship_matrix(dominant_clusters, localview_topk_indices).to(args.device)
        fuse2 = self.inter_channel_gcn(local_view1, Hcoarse, local_global_relationship_matrix)
        
        union_nodes = torch.unique(torch.cat((feature_topk_indices, localview_topk_indices), 0)).to(args.device)
        # Form a mask for edges that are between nodes in the top-k set
        mask_source = torch.any(edge_index[0].unsqueeze(-1) == union_nodes.unsqueeze(0), dim=-1)
        mask_target = torch.any(edge_index[1].unsqueeze(-1) == union_nodes.unsqueeze(0), dim=-1)

        # Combine the two masks with logical AND operation
        edge_mask = mask_source & mask_target
        edge_mask = edge_mask
        edge_index_topk = edge_index[:, edge_mask]

        # Create a mapping from old node indices to new node indices
        sorted_topk_indices, new_indices = union_nodes.sort()
        node_mapping = torch.full((x.size(0),), -1, dtype=torch.long).to(args.device)
        node_mapping[sorted_topk_indices] = new_indices

        # Remap node indices in edge_index
        edge_index_topk = node_mapping[edge_index_topk]

        # Remove edges that point to non-existent nodes (-1)
        edge_mask = (edge_index_topk[0] != -1) & (edge_index_topk[1] != -1)
        edge_index_topk = edge_index_topk[:, edge_mask]
        
        Fp = torch.zeros(len(union_nodes), fuse1.size(1)).to(args.device) 
        # Update Xp based on indices
        for i, idx in enumerate(union_nodes):
            if idx in feature_topk_indices and idx in localview_topk_indices:
                Fp[i] = (fuse2[torch.where(feature_topk_indices == idx)[0]] + fuse1[torch.where(localview_topk_indices == idx)[0]]) / 2
            elif idx in feature_topk_indices:
                Fp[i] = fuse2[torch.where(feature_topk_indices == idx)[0]]
            else:
                Fp[i] = fuse1[torch.where(localview_topk_indices == idx)[0]]
        new_batch = batch[union_nodes]
        return Fp, edge_index_topk, new_batch
    
#Inter-channel GCN Block
class InterChannelGCN(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=True, normalize=False):
        super(InterChannelGCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.add_self = add_self
        self.normalize = normalize
        
        # Attention mechanism components
        self.attention = nn.Linear(input_dim * 2, 1)  # Compute attention scores
        
        # Transformation after attention
        self.transform = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.transform.weight)

    def forward(self, H_fine, H_coarse, inter_channel_adj):
        # Assuming inter_channel_adj is a sparse matrix indicating relationships
        
        # Create an expanded version of H_coarse that matches H_fine for broadcasting
        H_coarse_expanded = H_coarse.repeat(H_fine.size(0), 1, 1)
        
        # Concatenate H_fine with each H_coarse node for attention computation
        H_fine_expanded = H_fine.unsqueeze(1).expand(-1, H_coarse.size(0), -1)
        concat_features = torch.cat((H_fine_expanded, H_coarse_expanded), dim=-1)
        
        # Compute attention scores
        attention_scores = self.attention(concat_features).squeeze(-1)
        
        # Mask attention scores using inter_channel_adj
        masked_attention_scores = attention_scores * inter_channel_adj
        
        # Normalize the masked attention scores to obtain attention weights
        attention_weights = F.softmax(masked_attention_scores, dim=-1).unsqueeze(1)
        
        # Apply attention weights to H_coarse
        attention_output = torch.bmm(attention_weights, H_coarse_expanded).squeeze(1)
        
        if self.add_self:
            attention_output += H_fine
        
        # Apply transformation
        out = self.transform(attention_output)
        
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        
        return F.relu(out)