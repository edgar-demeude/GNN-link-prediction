# visualization.py

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.metrics import precision_recall_curve

def plot_training_curves(all_histories: list):
    """Plots the loss, AUC, and AP evolution for all training runs."""
    print("\n--- Plotting Training Curves ---")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, history in enumerate(all_histories):
        # Loss
        axes[0].plot(history['train_losses'], label=f'Train Run {i+1}', alpha=0.5, linewidth=0.8)
        axes[0].plot(history['val_losses'], label=f'Val Run {i+1}', alpha=0.5, linewidth=0.8, linestyle='--')
        
        # AUC
        axes[1].plot(history['val_aucs'], label=f'Run {i+1}', alpha=0.6, linewidth=0.8)
        
        # AP
        axes[2].plot(history['val_aps'], label=f'Run {i+1}', alpha=0.6, linewidth=0.8)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Evolution (All Runs)')
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC Score')
    axes[1].set_title('ROC-AUC on Validation (All Runs)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Average Precision')
    axes[2].set_title('Average Precision on Validation (All Runs)')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_reconstruction_analysis(Ahat, y_true_val, y_pred_val, best_threshold_f1, best_threshold_density):
    """
    Plots the Edge Score Distribution and the Precision-Recall Curve.
    
    Args:
        Ahat: Adjacency matrix of prediction scores (all-to-all).
        y_true_val: True labels for the validation set.
        y_pred_val: Predicted scores for the validation set.
        best_threshold_f1: Optimal threshold based on F1 score.
        best_threshold_density: Optimal threshold based on density.
    """
    print("\n--- Plotting Reconstruction Analysis ---")
    precision, recall, thresholds = precision_recall_curve(y_true_val, y_pred_val)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Histogram
    axes[0].hist(Ahat.flatten(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(best_threshold_f1, color='red', linestyle='--', linewidth=2, label=f'Val F1 Thres: {best_threshold_f1:.4f}')
    axes[0].axvline(best_threshold_density, color='green', linestyle='--', linewidth=2, label=f'Density Thres: {best_threshold_density:.4f}')
    axes[0].set_xlabel('Prediction Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Edge Score Distribution (Log Scale)')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall Curve
    axes[1].plot(recall, precision, color='blue', linewidth=2, label='PR Curve (Val Set)')
    # Mark the optimal F1 threshold point
    idx_f1 = np.argmax(f1_scores[:-1])
    axes[1].plot(recall[idx_f1], precision[idx_f1], 'ro', markersize=8, label=f'Max F1 @ {best_threshold_f1:.4f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve (Validation)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_geographical_comparison(G_original: nx.Graph, G_recon_best: nx.Graph):
    """Plots the geographical comparison of the original vs. reconstructed graph."""
    print("\n--- Plotting Geographical Comparison ---")
    
    # Extract geographic positions
    geo_pos = {}
    for node in G_original.nodes():
        if 'lon' in G_original.nodes[node] and 'lat' in G_original.nodes[node]:
            lon = G_original.nodes[node]['lon']
            lat = G_original.nodes[node]['lat']
            geo_pos[node] = (lon, lat)

    fig, axes = plt.subplots(1, 2, figsize=(24, 10), 
                             subplot_kw={'projection': ccrs.Robinson(central_longitude=-20)})

    for idx, (graph, title, edge_color) in enumerate([
        (G_original, f"Original Graph ({G_original.number_of_edges()} edges)", 'gray'),
        (G_recon_best, f"Reconstructed ({len(G_recon_best.edges())} edges)", 'orange')
    ]):
        ax = axes[idx]
        
        # Add map features
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.gridlines(draw_labels=False, alpha=0.3)
        
        # Draw edges
        for edge in graph.edges():
            if edge[0] in geo_pos and edge[1] in geo_pos:
                lon1, lat1 = geo_pos[edge[0]]
                lon2, lat2 = geo_pos[edge[1]]
                ax.plot([lon1, lon2], [lat1, lat2], 
                        color=edge_color, linewidth=0.3, alpha=0.4,
                        transform=ccrs.Geodetic())
        
        # Draw nodes
        nodes_to_plot = [n for n in graph.nodes() if n in geo_pos]
        if nodes_to_plot:
            lons = [geo_pos[node][0] for node in nodes_to_plot]
            lats = [geo_pos[node][1] for node in nodes_to_plot]
            ax.scatter(lons, lats, s=15, c='red', alpha=0.7, 
                       transform=ccrs.PlateCarree(), zorder=5)
        
        ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()
    plt.show()