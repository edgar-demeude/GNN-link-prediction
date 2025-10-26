import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import random
import scipy.stats as st
from torch_geometric.utils import negative_sampling

# --- Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int):
    """Sets seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_loss(model, z, pos_edge_index, num_nodes, temperature=1.0):
    """Calculates the combined BCE + KL loss with weight balancing."""
    
    pos_edge_index = pos_edge_index.long()
    pos_logits = model.decode(z, pos_edge_index, sigmoid=False) / temperature
    
    # Negative edge sampling (1:1 ratio)
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=pos_edge_index.size(1)
    )
    neg_logits = model.decode(z, neg_edge_index, sigmoid=False) / temperature

    pos_labels = torch.ones_like(pos_logits)
    neg_labels = torch.zeros_like(neg_logits)
    all_logits = torch.cat([pos_logits, neg_logits])
    all_labels = torch.cat([pos_labels, neg_labels])

    # Calculate real-world positive weight (sparse graph compensation)
    num_pos = pos_edge_index.size(1)

    # Total possible unique edges in an undirected graph
    num_possible_edges = num_nodes * (num_nodes - 1) // 2
    num_neg_real = num_possible_edges - num_pos

    # Calculate weight: ratio of true negatives to true positives, capped at 10.0
    pos_weight_value = min(num_neg_real / num_pos, 10.0) 
    pos_weight_tensor = torch.tensor(pos_weight_value, dtype=torch.float, device=all_logits.device) 
    
    # BCE Loss
    bce_loss = F.binary_cross_entropy_with_logits(
        all_logits,
        all_labels,
        pos_weight=pos_weight_tensor
    )

    # KL Divergence Loss
    kl_loss = (1 / num_nodes) * model.kl_loss()
    return bce_loss + 0.1 * kl_loss # beta = 0.1

def aggregate_stats(test_aucs: list, test_aps: list, n_runs: int, confidence_level=0.95):
    """Calculates mean, std, and confidence interval for AUC and AP."""
    
    mean_auc = np.mean(test_aucs)
    std_auc = np.std(test_aucs, ddof=1)
    mean_ap = np.mean(test_aps)
    std_ap = np.std(test_aps, ddof=1)
    
    # Calculate 95% CI margin (using the t-distribution factor 1.96 for large N or common practice)
    # Using the exact calculation for CI
    if n_runs > 1:
        ci_auc = st.t.interval(confidence_level, n_runs-1, loc=mean_auc, scale=st.sem(test_aucs))
        ci_ap = st.t.interval(confidence_level, n_runs-1, loc=mean_ap, scale=st.sem(test_aps))
        ci_auc_margin = (mean_auc - ci_auc[0])
        ci_ap_margin = (mean_ap - ci_ap[0])
    else:
        ci_auc_margin = 0.0
        ci_ap_margin = 0.0

    return {
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'ci_auc': ci_auc_margin, 
        'mean_ap': mean_ap,
        'std_ap': std_ap,
        'ci_ap': ci_ap_margin
    }

def reconstruct_graph(Ahat, G, threshold, method_name):
    """Reconstructs the graph using a given threshold and computes metrics."""
    G_reconstructed = nx.Graph()
    G_reconstructed.add_nodes_from(G.nodes(data=True))
    
    # Get upper triangular indices (excluding diagonal)
    i_idx, j_idx = np.triu_indices(Ahat.shape[0], k=1)
    nodes_list = list(G.nodes())
    # Add edge if score > threshold
    edges_to_add = [(nodes_list[i], nodes_list[j], Ahat[i, j])
                    for i, j in zip(i_idx, j_idx) if Ahat[i, j] > threshold]
    
    G_reconstructed.add_weighted_edges_from(edges_to_add)
    
    # Statistics
    original_edges = G.number_of_edges()
    predicted_edges = G_reconstructed.number_of_edges()
    original_set = set(map(frozenset, G.edges()))
    predicted_set = set(map(frozenset, G_reconstructed.edges()))
    common_edges = len(original_set & predicted_set)
    
    precision = 100 * common_edges / predicted_edges if predicted_edges > 0 else 0
    recall = 100 * common_edges / original_edges
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    print(f"\n📊 {method_name}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Original Edges: {original_edges}")
    print(f"  Predicted Edges: {predicted_edges}")
    print(f"  Common Edges: {common_edges} ({recall:.2f}% Recall)")
    print(f"  Precision: {precision:.2f}%")
    print(f"  F1-score: {f1:.4f}")
    
    metrics = {'precision': precision, 'recall': recall, 'f1': f1}
    return G_reconstructed, metrics, predicted_edges