import torch
from .utils import get_loss, set_seed
from sklearn.metrics import roc_auc_score, average_precision_score

def train_and_evaluate(run_seed: int, data, device, model, optimizer):
    """
    Trains the VGAE model for a single run and evaluates it.
    Returns test metrics and training history.
    """
    # 1. Set seeds for this run
    set_seed(run_seed)
    
    # 2. Define train function (closure over model, optimizer, data)
    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.train_pos_edge_index)
        # Pass model instance to get_loss for kl_loss()
        loss = get_loss(model, z, data.train_pos_edge_index, data.num_nodes) 
        loss.backward()
        optimizer.step()
        return loss.item()
    
    # 3. Define evaluate function (closure over model, data)
    def evaluate(pos_edge_index, neg_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(data.x, data.train_pos_edge_index)
            # Pass model instance to get_loss for val loss calculation
            loss = get_loss(model, z, pos_edge_index, data.num_nodes)
            
            pos_pred = model.decode(z, pos_edge_index, sigmoid=True)
            neg_pred = model.decode(z, neg_edge_index, sigmoid=True)
            
            y_true = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))]).cpu().numpy()
            y_pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()
            
            auc = roc_auc_score(y_true, y_pred)
            ap = average_precision_score(y_true, y_pred)
        
        return loss.item(), auc, ap
    
    # 4. Training loop initialization
    train_losses, val_losses = [], []
    val_aucs, val_aps = [], []
    
    best_ap = 0
    patience = 50
    patience_counter = 0
    best_model_state = model.state_dict().copy() # Initial state
    
    # 5. Training loop (Max 500 epochs)
    for epoch in range(1, 501):
        loss = train()
        train_losses.append(loss)
        
        val_loss, val_auc, val_ap = evaluate(data.val_pos_edge_index, data.val_neg_edge_index)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        val_aps.append(val_ap)
        
        # Early stopping logic
        if val_ap > best_ap:
            best_ap = val_ap
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Train: {loss:.4f} | Val Loss: {val_loss:.4f} | AUC: {val_auc:.4f} | AP: {val_ap:.4f} | Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(best_model_state)
            break
            
    # 6. Final test evaluation with best model
    test_loss, test_auc, test_ap = evaluate(data.test_pos_edge_index, data.test_neg_edge_index)
    
    return test_auc, test_ap, train_losses, val_losses, val_aucs, val_aps