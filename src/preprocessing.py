import networkx as nx
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import from_networkx, train_test_split_edges

def load_and_prepare_data(filepath: str, degree_threshold: int, device: torch.device):
    """
    Loads the graph, cleans it (removes low-degree nodes), 
    normalizes features, performs One-Hot Encoding, and prepares 
    the PyTorch Geometric Data object with edge splitting.
    
    Returns: data (PyG object), G (cleaned NetworkX graph), X_ohe_tensor_ref (OHE features tensor)
    """
    # 1. Graph Loading and Cleaning
    print("--- 1. Graph Loading and Cleaning ---")
    G = nx.read_graphml(filepath)
    G.graph = {}

    # Replace None with 0.0
    for n, attr in G.nodes(data=True):
        for key in ['population', 'lat', 'lon']:
            if attr.get(key) is None:
                attr[key] = 0.0

    # Recursive removal of low-degree nodes
    removed_total = 0
    while True:
        low_degree_nodes = [n for n, d in G.degree() if d <= degree_threshold]
        if not low_degree_nodes:
            break
        G.remove_nodes_from(low_degree_nodes)
        removed_total += len(low_degree_nodes)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    min_degree = min([d for n, d in G.degree()])
    avg_degree = sum([d for n, d in G.degree()]) / num_nodes
    density = (2 * num_edges) / (num_nodes * (num_nodes - 1))
    
    print(f"Remaining nodes after filtering (D<={degree_threshold}): {num_nodes}")
    print(f"Remaining edges: {num_edges}")
    print(f"Removed nodes: {removed_total}")
    print(f"Min degree: {min_degree}")
    print(f"Avg degree: {avg_degree:.2f}")
    print(f"Density: {density:.4f}")

    # 2. PyG Conversion and Feature Extraction
    data = from_networkx(G)
    node_attributes = nx.get_node_attributes(G, 'country')
    country_list = [node_attributes.get(n, 'UNKNOWN') for n in G.nodes()]

    numerical_data = []
    for node_id in G.nodes():
        attr = G.nodes[node_id]
        # Order: lon, lat, population
        numerical_data.append([attr['lon'], attr['lat'], attr['population']])
        
    numerical_data = np.array(numerical_data, dtype=np.float32)

    # 3. Normalization (Numerical) and OHE (Categorical)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numerical_data)
    X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float)

    country_series = pd.Series(country_list)
    country_ohe = pd.get_dummies(country_series, prefix='country')
    X_ohe_tensor = torch.tensor(country_ohe.values, dtype=torch.float)
    
    # Reference to OHE tensor for final summary printout
    X_ohe_tensor_ref = X_ohe_tensor.clone() 

    # Concatenate features
    data.x = torch.cat([X_scaled_tensor, X_ohe_tensor], dim=1).to(device)

    # 4. Train/Val/Test Edge Split
    print("\n--- 2. Edge Splitting and Finalization ---")
    data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.2)

    # Convert edge indices to long and move to device
    for key in ["train_pos_edge_index", "val_pos_edge_index", "val_neg_edge_index",
                "test_pos_edge_index", "test_neg_edge_index"]:
        if hasattr(data, key) and getattr(data, key) is not None:
            setattr(data, key, getattr(data, key).long().to(device))

    print(f"Training edges: {data.train_pos_edge_index.shape[1]}")
    print(f"Validation edges: {data.val_pos_edge_index.shape[1]}")
    print(f"Test edges: {data.test_pos_edge_index.shape[1]}")
    
    return data, G, X_ohe_tensor_ref