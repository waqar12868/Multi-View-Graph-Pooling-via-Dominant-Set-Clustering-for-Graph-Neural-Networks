# Multi-View-Graph-Pooling-via-Dominant-Set-Clustering-for-Graph-Neural-Networks
 Graph pooling is a crucial operation in Graph Neural Networks to perform graph classification tasks by reducing complexity and preserving structural information of the input graph. However, most existing pooling methods tend to overlook edge weights and employ a single-view pooling strategy, either looking at local or global topological information, that may not capture the graph's comprehensive structural information. To overcome these limitations, this study introduces the Dominant Set Multi-View Pooling (DSMVPool) method with two main contributions. First, we develop a novel dominant set cluster pooling method that analyzes the graph's overall architecture and connectivity patterns by finding the clusters and generating its pooled coarser view. Moreover, we generate two pooled graph views by extracting the most significant nodes based on the graph's local topological information and node features. Second, we design a fusion-view attention layer to fuse the coarser graph view with two pooled graph views, which allows our pooling method to extract rich discriminative global, local topological information with node features and edge weight simultaneously. Comprehensive experiments are performed on four different kinds of graph classification benchmarks, including computer vision, chemical, biological, and social network tasks, demonstrating superior performance against the state-of-the-art.
# Requirements
pytorch=1.13.1

torch-geometric=2.3.0

torch-scatter=2.1.1

torch-sparse=0.6.17

torch-spline-conv=1.2.2

torch-cluster=1.6.1
