import networkx as nx
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import from_networkx, train_test_split_edges
from sklearn.feature_extraction.text import TfidfVectorizer


def load_and_prepare_data(filepath: str, device: torch.device):
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
    for _, attr in G.nodes(data=True):
        for key in ['population', 'lat', 'lon']:
            if attr.get(key) is None:
                attr[key] = 0.0

    # Filtering

    # Classic degree Filter (Threshold=5)
    G = _filter_graph_by_threshold(G.copy(), threshold=5)


    """
    Filters to try

    # Classic degree Filter (Threshold=5)
    G = _filter_graph_by_threshold(G.copy(), threshold=5)
    
    # TF-IDF (Text Focus: 70% kept)
    G = _filter_graph_by_tfidf(G.copy(), tfidf_weight=0.7, degree_weight=0.3, keep_percentile=70)

    # TF-IDF (Hub Focus: 70% kept)
    G = _filter_graph_by_tfidf(G.copy(), tfidf_weight=0.4, degree_weight=0.6, keep_percentile=70)
    
    # Random Walk (80% kept)
    G = _filter_graph_by_random_walk(G.copy(), importance_weight=0.8, keep_percentile=80)
    """
    
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



def _filter_graph_by_threshold(G, threshold=5):
    """Recursively removes nodes of low degree."""
    G_filtered = G.copy()
    removed_total = 0
    while True:
        low_degree_nodes = [n for n, d in G_filtered.degree() if d <= threshold]
        if not low_degree_nodes:
            break
        G_filtered.remove_nodes_from(low_degree_nodes)
        removed_total += len(low_degree_nodes)

    print(f"[Threshold Filter] Removed {removed_total} nodes (Degree <= {threshold}). Remaining: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")
    return G_filtered


def _filter_graph_by_tfidf(G,
                          text_fields=['country', 'name', 'city'],
                          tfidf_weight=0.7,
                          degree_weight=0.3,
                          keep_percentile=80,
                          stop_words='english'):
    """Remove low-importance nodes based on a TF-IDF + degree score."""

    node_ids = list(G.nodes())
    corpus = []
    for n in node_ids:
        attr = G.nodes[n]
        text = " ".join(str(attr.get(field, "")) for field in text_fields)
        corpus.append(text.strip())

    # Compute TF-IDF importance
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=1)).flatten()

    # Compute normalized degree importance
    degrees = np.array([G.degree(n) for n in node_ids], dtype=float)
    deg_min = degrees.min() if len(degrees) > 0 else 0
    deg_max = degrees.max() if len(degrees) > 0 else 0
    deg_norm = (degrees - deg_min) / (deg_max - deg_min + 1e-8)

    # Combined node importance
    combined_score = tfidf_weight * tfidf_scores + degree_weight * deg_norm

    # Keep only top percentile
    threshold_value = np.percentile(combined_score, 100 - keep_percentile)
    nodes_to_keep = [node_ids[i] for i, s in enumerate(combined_score) if s >= threshold_value]

    G_filtered = G.subgraph(nodes_to_keep).copy()

    removed_nodes = len(G.nodes()) - len(G_filtered.nodes())
    print(f"[TF-IDF Filter] Removed {removed_nodes} nodes (Keep Top {keep_percentile}%). Remaining: {len(G_filtered.nodes())} nodes, {len(G_filtered.edges())} edges")
    return G_filtered


def _filter_graph_by_random_walk(G, importance_weight=0.8, keep_percentile=70):
    """
    Filters nodes based on a centrality measure (e.g., Personalized PageRank, here approximated
    by a simple degree-weighted score) combined with edge weights.
    """
    node_ids = list(G.nodes())
    degrees = np.array([G.degree(n) for n in node_ids], dtype=float)

    # Normalize degree
    deg_min, deg_max = degrees.min(), degrees.max()
    if deg_max == deg_min:
        normalized_scores = np.ones_like(degrees)
    else:
        normalized_scores = (degrees - deg_min) / (deg_max - deg_min)

    # Combine with a random score for simple approximation of "random walk" importance
    random_factor = np.random.rand(len(node_ids))
    combined_score = importance_weight * normalized_scores + (1 - importance_weight) * random_factor

    # Keep only top percentile
    threshold_value = np.percentile(combined_score, 100 - keep_percentile)
    nodes_to_keep = [node_ids[i] for i, s in enumerate(combined_score) if s >= threshold_value]

    G_filtered = G.subgraph(nodes_to_keep).copy()

    removed_nodes = len(G.nodes()) - len(G_filtered.nodes())
    print(f"[Random Walk Filter] Removed {removed_nodes} nodes (Keep Top {keep_percentile}%). Remaining: {len(G_filtered.nodes())} nodes, {len(G_filtered.edges())} edges")
    return G_filtered
