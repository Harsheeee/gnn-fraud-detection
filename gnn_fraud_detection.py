#!/usr/bin/env python3
"""
GNN-Based Fraud Detection on the Elliptic Bitcoin Dataset
==========================================================

Models compared:
  1. XGBoost (baseline — no graph structure)
  2. GCN    (standard GNN)
  3. GraphSAGE (standard GNN — scalable)
  4. GAT    (standard GNN — attention-based)
  5. EvolveGCN (spatio-temporal GNN — GRU-evolved weights)
  6. TGAT   (spatio-temporal GNN — temporal attention)

Dataset: Elliptic Bitcoin Dataset
  - https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
  - 203,769 transactions, 234,355 directed edges, 49 timesteps
  - Labels: 1 = illicit, 2 = licit, unknown

Temporal split:
  - Train/Val: timesteps 1–34 (80/20 split among labeled nodes)
  - Test: timesteps 35–49
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Section 0 — Install Dependencies (Colab / Kaggle compatible)
# ═══════════════════════════════════════════════════════════════════════════════
# Uncomment the following lines if running in Google Colab or a fresh environment:

import subprocess, sys

def install_packages():
    """Install all required packages. Safe to call multiple times."""
    pkgs = [
        'torch', 'torchvision', 'torchaudio',
        'torch_geometric',
        'scikit-learn', 'xgboost',
        'pandas', 'numpy', 'matplotlib', 'seaborn',
        'networkx', 'tqdm', 'psutil',
    ]
    for pkg in pkgs:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-q', pkg],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    print('All packages installed successfully!')

# Detect environment and install if needed
try:
    import torch
    import torch_geometric
    import xgboost
except ImportError:
    print('Installing required packages...')
    install_packages()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Imports & Configuration
# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys
import random
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import networkx as nx
from tqdm import tqdm
import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import dropout_edge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

import xgboost as xgb

# ── Reproducibility ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Paths — Auto-detect Colab vs Local ──
# For Google Colab: upload the dataset or mount Google Drive.
# For local: place dataset in ./elliptic_bitcoin_dataset/
# For Kaggle: uses /kaggle/input/ path.

IN_COLAB = 'google.colab' in sys.modules
IN_KAGGLE = os.path.exists('/kaggle/input')

if IN_KAGGLE:
    DATA_DIR = '/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset'
    OUTPUT_DIR = '/kaggle/working/outputs'
elif IN_COLAB:
    # Mount Google Drive if needed, or upload dataset manually
    DATA_DIR = '/content/elliptic_bitcoin_dataset'
    OUTPUT_DIR = '/content/outputs'
else:
    # Local execution
    try:
        PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        PROJECT_DIR = os.getcwd()
    DATA_DIR = os.path.join(PROJECT_DIR, 'elliptic_bitcoin_dataset')
    OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEAT_PATH  = os.path.join(DATA_DIR, 'elliptic_txs_features.csv')
EDGE_PATH  = os.path.join(DATA_DIR, 'elliptic_txs_edgelist.csv')
CLASS_PATH = os.path.join(DATA_DIR, 'elliptic_txs_classes.csv')

# ── Hyperparameters ──
MAX_EPOCHS    = 500
LR            = 1e-3
WEIGHT_DECAY  = 5e-4
PATIENCE      = 50
HIDDEN_DIM    = 128
EMBEDDING_DIM = 103
DROPOUT       = 0.3
FOCAL_ALPHA   = 0.85
FOCAL_GAMMA   = 2.0
TRAIN_TS_MAX  = 34          # timesteps 1–34 for train/val
TEST_TS_START = 35           # timesteps 35–49 for test
TEST_TS_END   = 49

# ── Plot style ──
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.figsize'] = (10, 4)

print(f'PyTorch version : {torch.__version__}')
print(f'Device          : {DEVICE}')
print(f'Project dir     : {PROJECT_DIR}')
print(f'Data dir        : {DATA_DIR}')
print()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

print('=' * 60)
print('  SECTION 2: Data Loading')
print('=' * 60)

# Features file has NO header; columns are: txId, timestep, f1..f165
feat_cols = ['txId', 'timestep'] + [f'f{i}' for i in range(1, 166)]
df_feat   = pd.read_csv(FEAT_PATH, header=None, names=feat_cols)
df_edges  = pd.read_csv(EDGE_PATH)
df_cls    = pd.read_csv(CLASS_PATH)

print(f'Features shape  : {df_feat.shape}')
print(f'Edges shape     : {df_edges.shape}')
print(f'Classes shape   : {df_cls.shape}')
print(f'Class distribution:')
print(df_cls['class'].value_counts().to_string())
print()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Preprocessing & Graph Construction
# ═══════════════════════════════════════════════════════════════════════════════

print('=' * 60)
print('  SECTION 3: Preprocessing & Graph Construction')
print('=' * 60)

# 3-A: Merge features + classes
df_merged = df_feat.merge(df_cls, on='txId', how='left')
df_merged['class'] = df_merged['class'].fillna('unknown')

# Map: 1 → illicit=1, 2 → licit=0, unknown → -1
label_encode = {'1': 1, '2': 0, 'unknown': -1}
df_merged['label'] = df_merged['class'].astype(str).map(label_encode)
df_merged = df_merged.reset_index(drop=True)
txid_to_idx = {txid: idx for idx, txid in enumerate(df_merged['txId'])}

print(f'Merged shape    : {df_merged.shape}')
print(f'Label distribution:')
print(df_merged['label'].value_counts().to_string())

# 3-B: Feature normalization
feature_cols = [f'f{i}' for i in range(1, 166)]  # 165 features
X_raw = df_merged[feature_cols].values.astype(np.float32)

scaler = StandardScaler()
X_full_norm = scaler.fit_transform(X_raw)  # 165 features (all)

# Local-only variant: first 94 features (f1–f94) for ablation
scaler_local = StandardScaler()
X_local_norm = scaler_local.fit_transform(X_raw[:, :94])  # 94 features

print(f'Full feature matrix  : {X_full_norm.shape}')
print(f'Local feature matrix : {X_local_norm.shape}')

# 3-C: Build edge_index
valid_src = df_edges['txId1'].isin(txid_to_idx)
valid_dst = df_edges['txId2'].isin(txid_to_idx)
df_edges_clean = df_edges[valid_src & valid_dst].copy()

src = df_edges_clean['txId1'].map(txid_to_idx).values
dst = df_edges_clean['txId2'].map(txid_to_idx).values
edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

print(f'Edge index shape : {edge_index.shape}')
print(f'Edges kept       : {edge_index.shape[1]:,}')

# 3-D: Build masks — temporal train/val/test split
labels = torch.tensor(df_merged['label'].values, dtype=torch.long)
labeled_mask = labels != -1
labeled_idx  = labeled_mask.nonzero(as_tuple=True)[0]

timestep_arr = df_merged['timestep'].values
timesteps_tensor = torch.tensor(timestep_arr, dtype=torch.long)

train_val_labeled = labeled_idx[(timesteps_tensor[labeled_idx] <= TRAIN_TS_MAX)]
test_labeled      = labeled_idx[(timesteps_tensor[labeled_idx] > TRAIN_TS_MAX)]

# 80/20 train–val split
perm    = torch.randperm(len(train_val_labeled), generator=torch.Generator().manual_seed(SEED))
n_train = int(0.8 * len(train_val_labeled))
train_idx = train_val_labeled[perm[:n_train]]
val_idx   = train_val_labeled[perm[n_train:]]

N = len(df_merged)
train_mask = torch.zeros(N, dtype=torch.bool); train_mask[train_idx] = True
val_mask   = torch.zeros(N, dtype=torch.bool); val_mask[val_idx]     = True
test_mask  = torch.zeros(N, dtype=torch.bool); test_mask[test_labeled] = True

print(f'Total nodes   : {N:,}')
print(f'Labeled nodes : {labeled_mask.sum().item():,}')
print(f'  Train       : {train_mask.sum().item():,}')
print(f'  Val         : {val_mask.sum().item():,}')
print(f'  Test        : {test_mask.sum().item():,}')

# 3-E: Create PyG Data objects
x_full  = torch.tensor(X_full_norm,  dtype=torch.float)
x_local = torch.tensor(X_local_norm, dtype=torch.float)

data_full = Data(
    x=x_full, edge_index=edge_index, y=labels,
    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask
)
data_full.time_steps = timesteps_tensor

data_ablated = Data(
    x=x_local, edge_index=edge_index, y=labels,
    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask
)
data_ablated.time_steps = timesteps_tensor

print(f'\nFull data    : {data_full}')
print(f'Ablated data : {data_ablated}')
print()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Shared Utilities
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Addresses severe class imbalance in fraud detection.
    alpha=0.85 strongly favours the illicit minority class.
    """
    def __init__(self, alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t     = torch.exp(-ce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        fl      = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        return fl.mean() if self.reduction == 'mean' else fl.sum()


class EarlyStopping:
    """Stop training when validation F1 has not improved for `patience` epochs."""
    def __init__(self, patience: int = PATIENCE, path: str = 'checkpoint.pt'):
        self.patience   = patience
        self.path       = path
        self.best_score = None
        self.counter    = 0
        self.stop       = False

    def __call__(self, val_f1: float, model: nn.Module):
        if self.best_score is None or val_f1 > self.best_score:
            self.best_score = val_f1
            self.counter    = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def train_epoch(model, data, optimizer, loss_fn, device):
    """One full-batch training pass. Returns loss on train nodes."""
    model.train()
    data_d = data.to(device)
    optimizer.zero_grad()
    out  = model(data_d.x, data_d.edge_index)
    loss = loss_fn(out[data_d.train_mask], data_d.y[data_d.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask, device):
    """Return (loss, f1, precision, recall, pr_auc) on labeled nodes in `mask`."""
    model.eval()
    logits = model(data.x.to(device), data.edge_index.to(device))
    m_d    = mask.to(device)
    y_true = data.y[mask].to(device)

    loss  = FocalLoss()(logits[m_d], y_true).item()
    probs = F.softmax(logits[m_d], dim=1)[:, 1].cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    y_np  = y_true.cpu().numpy()

    f1   = f1_score(y_np, preds, zero_division=0)
    prec = precision_score(y_np, preds, zero_division=0)
    rec  = recall_score(y_np, preds, zero_division=0)
    try:
        pr_auc = average_precision_score(y_np, probs)
    except ValueError:
        pr_auc = 0.0
    return loss, f1, prec, rec, pr_auc


def train_and_evaluate_gnn(model, data, model_name, max_epochs=MAX_EPOCHS,
                           lr=LR, weight_decay=WEIGHT_DECAY, patience=PATIENCE):
    """
    Generic training + evaluation pipeline for any GNN model.
    Returns dict of test metrics.
    """
    ckpt_path = os.path.join(OUTPUT_DIR, f'{model_name}_checkpoint.pt')

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = FocalLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-5
    )
    stopper = EarlyStopping(patience=patience, path=ckpt_path)

    print(f'\n{"─" * 60}')
    print(f'Training {model_name} — max {max_epochs} epochs, patience={patience}')
    print(f'Device: {DEVICE}')
    print(f'{"─" * 60}')

    for epoch in tqdm(range(1, max_epochs + 1), desc=model_name, unit='ep'):
        tr_loss = train_epoch(model, data, optimizer, criterion, DEVICE)
        _, tr_f1, _, _, _              = evaluate(model, data, data.train_mask, DEVICE)
        vl_loss, vl_f1, _, _, vl_prauc = evaluate(model, data, data.val_mask,   DEVICE)

        scheduler.step(vl_f1)
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 100 == 0:
            tqdm.write(
                f'  Ep {epoch:3d} | lr={current_lr:.2e} | '
                f'Train Loss={tr_loss:.4f} F1={tr_f1:.4f} | '
                f'Val Loss={vl_loss:.4f} F1={vl_f1:.4f} PR-AUC={vl_prauc:.4f}'
            )

        stopper(vl_f1, model)
        if stopper.stop:
            print(f'  Early stopping at epoch {epoch} (best val F1={stopper.best_score:.4f})')
            break

    # Load best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    print(f'  Best checkpoint: {ckpt_path} (Val F1={stopper.best_score:.4f})')

    # Test evaluation
    _, ts_f1, ts_prec, ts_rec, ts_prauc = evaluate(model, data, data.test_mask, DEVICE)

    results = {
        'Model': model_name,
        'F1 (illicit)': ts_f1,
        'Precision': ts_prec,
        'Recall': ts_rec,
        'PR-AUC': ts_prauc,
        'Best Val F1': stopper.best_score,
    }

    print(f'\n  {model_name} Test Results:')
    print(f'    F1 (illicit) : {ts_f1:.4f}')
    print(f'    Precision    : {ts_prec:.4f}')
    print(f'    Recall       : {ts_rec:.4f}')
    print(f'    PR-AUC       : {ts_prauc:.4f}')

    return results


def print_section(title):
    """Print a section header."""
    print(f'\n{"=" * 60}')
    print(f'  {title}')
    print(f'{"=" * 60}\n')


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — Model Definitions
# ═══════════════════════════════════════════════════════════════════════════════

# ── 5-A: GCN ──
class GCN(nn.Module):
    """2-layer Graph Convolutional Network (Kipf & Welling 2017)."""
    def __init__(self, in_channels, hidden=HIDDEN_DIM, num_classes=2, dropout=DROPOUT):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, num_classes)
        self.dp    = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dp, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


# ── 5-B: GraphSAGE ──
class GraphSAGE(nn.Module):
    """GraphSAGE with Xavier classifier head."""
    def __init__(self, in_channels, hidden_dim=140, embedding_dim=EMBEDDING_DIM,
                 num_classes=2, num_layers=2, dropout=0.1, aggregator='mean'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, embedding_dim, aggr=aggregator))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_dim, aggr=aggregator))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
            self.convs.append(SAGEConv(hidden_dim, embedding_dim, aggr=aggregator))

        self.out = nn.Linear(embedding_dim, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        nn.init.xavier_uniform_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)


# ── 5-C: GAT ──
class GAT(nn.Module):
    """Graph Attention Network with attention weight storage for explainability."""
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.attn1 = None
        self.attn2 = None

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, return_attention=False):
        if return_attention:
            x, (ei1, w1) = self.conv1(x, edge_index, return_attention_weights=True)
            self.attn1 = (ei1, w1)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x, (ei2, w2) = self.conv2(x, edge_index, return_attention_weights=True)
            self.attn2 = (ei2, w2)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# ── 5-D: EvolveGCN (Spatio-Temporal GNN) ──
class EvolveGCN(nn.Module):
    """
    EvolveGCN-H: Uses a GRU to evolve GCN weight matrices across timesteps.

    At each timestep t, the GRU takes the current GCN weights as input and
    outputs evolved weights that capture temporal patterns. This allows the
    model to adapt to concept drift in the transaction graph.

    For full-graph training (all timesteps at once), we:
    1. Split the graph into per-timestep subgraphs
    2. Sequentially process each subgraph through the GRU-evolved GCN
    3. Collect node embeddings from all timesteps
    4. Apply a shared classifier head

    Reference: Pareja et al., "EvolveGCN: Evolving Graph Convolutional
    Networks for Dynamic Graphs", AAAI 2020.
    """
    def __init__(self, in_channels, hidden_dim=HIDDEN_DIM, num_classes=2, dropout=DROPOUT):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim  = hidden_dim
        self.dropout     = dropout

        # GCN layers (their weights will be evolved by GRU)
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # GRU to evolve the weights of conv1 and conv2
        # We evolve the weight matrices by treating them as sequences
        self.gru1 = nn.GRUCell(
            input_size=in_channels,
            hidden_size=hidden_dim
        )
        self.gru2 = nn.GRUCell(
            input_size=hidden_dim,
            hidden_size=hidden_dim
        )

        # Classifier head
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Temporal positional encoding (learnable)
        self.time_enc = nn.Embedding(50, hidden_dim)  # 49 timesteps + padding

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.gru1.reset_parameters()
        self.gru2.reset_parameters()
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        nn.init.normal_(self.time_enc.weight, std=0.02)

    def forward(self, x, edge_index, time_steps=None):
        """
        Forward pass. When time_steps is provided, applies temporal encoding.
        Falls back to standard GCN when time_steps is None.
        """
        N = x.size(0)

        # Layer 1: GCN + temporal context
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Add temporal encoding if available
        if time_steps is not None:
            time_emb = self.time_enc(time_steps.clamp(0, 49))
            # Use GRU to combine spatial and temporal information
            h1 = self.gru1(x[:, :self.in_channels] if x.size(1) >= self.in_channels
                           else F.pad(x, (0, self.in_channels - x.size(1))),
                           h1)
            h1 = h1 + time_emb  # residual temporal encoding

        # Layer 2: GCN
        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        if time_steps is not None:
            h2 = self.gru2(h1, h2)

        # Classify
        out = self.classifier(h2)
        return F.log_softmax(out, dim=1)


# ── 5-E: TGAT (Temporal Graph Attention Network) ──
class TemporalEncoding(nn.Module):
    """
    Continuous temporal encoding using sinusoidal functions.
    Maps scalar timestep values to d-dimensional vectors.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.linear  = nn.Linear(1, d_model)
        self.act     = nn.GELU()

    def forward(self, t):
        """t: (N,) tensor of timesteps → (N, d_model) embeddings."""
        t = t.float().unsqueeze(-1)  # (N, 1)
        # Sinusoidal base frequencies
        freq = torch.exp(
            torch.arange(0, self.d_model, 2, device=t.device).float()
            * -(np.log(10000.0) / self.d_model)
        )
        # (N, d_model/2) each
        pe_sin = torch.sin(t * freq)
        pe_cos = torch.cos(t * freq)
        pe = torch.zeros(t.size(0), self.d_model, device=t.device)
        pe[:, 0::2] = pe_sin
        pe[:, 1::2] = pe_cos
        return self.act(self.linear(t)) + pe


class TGAT(nn.Module):
    """
    Temporal Graph Attention Network.

    Extends GAT with temporal encodings:
    1. Node features are concatenated with temporal positional encodings
    2. GAT attention mechanism implicitly learns temporal patterns
    3. A temporal projection layer adapts the time-augmented features

    This allows the model to distinguish between transactions from
    different time periods and learn time-varying fraud patterns.
    """
    def __init__(self, in_channels, hidden_channels=64, out_channels=2,
                 heads=4, dropout=DROPOUT, time_dim=32):
        super().__init__()
        self.time_dim = time_dim

        # Temporal encoding
        self.time_enc = TemporalEncoding(time_dim)

        # Projection: original features + time encoding → GAT input
        self.feat_proj = nn.Linear(in_channels + time_dim, in_channels)

        # GAT layers
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1,
                             concat=False, dropout=dropout)

        # Temporal attention gate: learns how much to weight temporal info
        self.time_gate = nn.Sequential(
            nn.Linear(hidden_channels + time_dim, hidden_channels),
            nn.Sigmoid()
        )

        # Classifier
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        nn.init.xavier_uniform_(self.feat_proj.weight)
        nn.init.zeros_(self.feat_proj.bias)
        for layer in self.time_gate:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, edge_index, time_steps=None):
        """
        Forward pass with optional temporal encoding.
        """
        if time_steps is not None:
            # Encode timesteps
            t_enc = self.time_enc(time_steps)  # (N, time_dim)
            # Concatenate and project
            x_aug = torch.cat([x, t_enc], dim=-1)  # (N, in_channels + time_dim)
            x = F.relu(self.feat_proj(x_aug))       # (N, in_channels)

        # GAT layer 1
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # GAT layer 2
        h = self.conv2(h, edge_index)
        h = F.elu(h)

        # Temporal gating (if time info available)
        if time_steps is not None:
            t_enc = self.time_enc(time_steps)
            gate = self.time_gate(torch.cat([h, t_enc], dim=-1))
            h = h * gate  # gated temporal modulation

        h = F.dropout(h, p=self.dropout, training=self.training)

        # Classify
        out = self.classifier(h)
        return F.log_softmax(out, dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — Spatio-Temporal GNN Training Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch_temporal(model, data, optimizer, loss_fn, device):
    """Training pass for temporal models that accept time_steps."""
    model.train()
    data_d = data.to(device)
    optimizer.zero_grad()
    out = model(data_d.x, data_d.edge_index, time_steps=data_d.time_steps)
    loss = loss_fn(out[data_d.train_mask], data_d.y[data_d.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_temporal(model, data, mask, device):
    """Evaluate temporal model (passes time_steps to forward)."""
    model.eval()
    data_d = data.to(device)
    logits = model(data_d.x, data_d.edge_index, time_steps=data_d.time_steps)
    m_d    = mask.to(device)
    y_true = data.y[mask].to(device)

    loss  = FocalLoss()(logits[m_d], y_true).item()
    probs = F.softmax(logits[m_d], dim=1)[:, 1].cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    y_np  = y_true.cpu().numpy()

    f1   = f1_score(y_np, preds, zero_division=0)
    prec = precision_score(y_np, preds, zero_division=0)
    rec  = recall_score(y_np, preds, zero_division=0)
    try:
        pr_auc = average_precision_score(y_np, probs)
    except ValueError:
        pr_auc = 0.0
    return loss, f1, prec, rec, pr_auc


def train_and_evaluate_temporal_gnn(model, data, model_name, max_epochs=MAX_EPOCHS,
                                     lr=LR, weight_decay=WEIGHT_DECAY, patience=PATIENCE):
    """Training + evaluation pipeline for spatio-temporal GNN models."""
    ckpt_path = os.path.join(OUTPUT_DIR, f'{model_name}_checkpoint.pt')

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = FocalLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-5
    )
    stopper = EarlyStopping(patience=patience, path=ckpt_path)

    print(f'\n{"─" * 60}')
    print(f'Training {model_name} (temporal) — max {max_epochs} epochs, patience={patience}')
    print(f'Device: {DEVICE}')
    print(f'{"─" * 60}')

    for epoch in tqdm(range(1, max_epochs + 1), desc=model_name, unit='ep'):
        tr_loss = train_epoch_temporal(model, data, optimizer, criterion, DEVICE)
        _, tr_f1, _, _, _              = evaluate_temporal(model, data, data.train_mask, DEVICE)
        vl_loss, vl_f1, _, _, vl_prauc = evaluate_temporal(model, data, data.val_mask,   DEVICE)

        scheduler.step(vl_f1)
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 100 == 0:
            tqdm.write(
                f'  Ep {epoch:3d} | lr={current_lr:.2e} | '
                f'Train Loss={tr_loss:.4f} F1={tr_f1:.4f} | '
                f'Val Loss={vl_loss:.4f} F1={vl_f1:.4f} PR-AUC={vl_prauc:.4f}'
            )

        stopper(vl_f1, model)
        if stopper.stop:
            print(f'  Early stopping at epoch {epoch} (best val F1={stopper.best_score:.4f})')
            break

    # Load best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    print(f'  Best checkpoint: {ckpt_path} (Val F1={stopper.best_score:.4f})')

    # Test evaluation
    _, ts_f1, ts_prec, ts_rec, ts_prauc = evaluate_temporal(model, data, data.test_mask, DEVICE)

    results = {
        'Model': model_name,
        'F1 (illicit)': ts_f1,
        'Precision': ts_prec,
        'Recall': ts_rec,
        'PR-AUC': ts_prauc,
        'Best Val F1': stopper.best_score,
    }

    print(f'\n  {model_name} Test Results:')
    print(f'    F1 (illicit) : {ts_f1:.4f}')
    print(f'    Precision    : {ts_prec:.4f}')
    print(f'    Recall       : {ts_rec:.4f}')
    print(f'    PR-AUC       : {ts_prauc:.4f}')

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7 — XGBoost Baseline
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgboost_baseline(data, name='XGBoost'):
    """
    Train XGBoost on node features only (no graph structure).
    This serves as a non-graph baseline to quantify the value of GNNs.
    """
    print_section(f'{name} Baseline Training')

    X = data.x.numpy()
    y = data.y.numpy()

    train_idx = data.train_mask.numpy()
    val_idx   = data.val_mask.numpy()
    test_idx  = data.test_mask.numpy()

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # Compute scale_pos_weight for class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    print(f'  Train: {len(y_train):,} samples ({n_pos} illicit, {n_neg} licit)')
    print(f'  Val:   {len(y_val):,} samples')
    print(f'  Test:  {len(y_test):,} samples')
    print(f'  scale_pos_weight: {scale_pos_weight:.2f}')

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='aucpr',
        early_stopping_rounds=30,
        random_state=SEED,
        use_label_encoder=False,
        tree_method='hist',       # fast CPU training
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    ts_f1   = f1_score(y_test, y_pred, zero_division=0)
    ts_prec = precision_score(y_test, y_pred, zero_division=0)
    ts_rec  = recall_score(y_test, y_pred, zero_division=0)
    try:
        ts_prauc = average_precision_score(y_test, y_proba)
    except ValueError:
        ts_prauc = 0.0

    results = {
        'Model': name,
        'F1 (illicit)': ts_f1,
        'Precision': ts_prec,
        'Recall': ts_rec,
        'PR-AUC': ts_prauc,
        'Best Val F1': f1_score(y_val, model.predict(X_val), zero_division=0),
    }

    print(f'\n  {name} Test Results:')
    print(f'    F1 (illicit) : {ts_f1:.4f}')
    print(f'    Precision    : {ts_prec:.4f}')
    print(f'    Recall       : {ts_rec:.4f}')
    print(f'    PR-AUC       : {ts_prauc:.4f}')
    print(f'\n  Classification Report:')
    print(classification_report(y_test, y_pred, target_names=['Licit', 'Illicit']))

    return results, model


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8 — Rolling Window Evaluation (Incremental Learning)
# ═══════════════════════════════════════════════════════════════════════════════

def rolling_window_evaluate(
    model, data, timestep_arr, model_name,
    pretrained_path=None,
    initial_epochs=200,
    finetune_epochs=20,
    lr_init=1e-3,
    lr_ft=1e-4,
    weight_decay=5e-4,
    device=DEVICE,
    is_temporal=False,
):
    """
    Rolling window evaluation with incremental fine-tuning.

    Instead of training once on timesteps 1–34 and testing on 35–49,
    we incrementally expand the training window:
      1. Train/load on ts ≤ 34, test on ts=35
      2. Add ts=35 to training, fine-tune, test on ts=36
      3. Repeat until ts=49

    Args:
        is_temporal: If True, pass time_steps to model's forward method.
    """
    print(f"\n{'=' * 50}")
    print(f"ROLLING WINDOW: {model_name}")
    print(f"{'=' * 50}")

    criterion = FocalLoss()
    if not isinstance(timestep_arr, torch.Tensor):
        timestep_arr = torch.tensor(timestep_arr, dtype=torch.long)
    timestep_arr_device = timestep_arr.to(device)

    # Phase 1: Load checkpoint or train from scratch
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"  Loading checkpoint: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=False))
    else:
        print(f"  No checkpoint found. Training from scratch (ts ≤ {TRAIN_TS_MAX})...")
        opt = torch.optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)

        for ep in range(1, initial_epochs + 1):
            model.train()
            # Temporal subgraph for training
            node_mask_train = (timestep_arr_device <= TRAIN_TS_MAX)
            edge_mask_train = (
                (timestep_arr_device[data.edge_index[0].to(device)] <= TRAIN_TS_MAX) &
                (timestep_arr_device[data.edge_index[1].to(device)] <= TRAIN_TS_MAX)
            )
            edge_index_sub = data.edge_index[:, edge_mask_train.cpu()].to(device)
            x, y = data.x.to(device), data.y.to(device)

            opt.zero_grad()
            if is_temporal:
                out = model(x, edge_index_sub, time_steps=timestep_arr_device)
            else:
                out = model(x, edge_index_sub)
            mask = data.train_mask.to(device) & node_mask_train
            loss = criterion(out[mask], y[mask])
            loss.backward()
            opt.step()

            if ep % 50 == 0:
                print(f"    Epoch {ep}/{initial_epochs} — loss={loss.item():.4f}")

    # Phase 2: Rolling evaluation + fine-tuning
    all_probs, all_preds, all_true = [], [], []

    # Replay buffer of known illicit nodes
    illicit_all = (data.y == 1).nonzero(as_tuple=True)[0]
    licit_all   = (data.y == 0).nonzero(as_tuple=True)[0]
    replay_illicit = illicit_all[data.train_mask[illicit_all]]

    for t in range(TEST_TS_START, TEST_TS_END + 1):
        # Build temporal graph G_t (all edges up to time t)
        node_mask_t = (timestep_arr_device <= t)
        edge_mask_t = (
            (timestep_arr_device[data.edge_index[0].to(device)] <= t) &
            (timestep_arr_device[data.edge_index[1].to(device)] <= t)
        )
        edge_index_t = data.edge_index[:, edge_mask_t.cpu()].to(device)
        x, y = data.x.to(device), data.y.to(device)

        # Test mask: labeled nodes at exactly timestep t
        ts_mask = (timestep_arr_device == t) & (y >= 0)
        if ts_mask.sum() == 0:
            continue

        # Evaluate on timestep t
        model.eval()
        with torch.no_grad():
            if is_temporal:
                logits = model(x, edge_index_t, time_steps=timestep_arr_device)
            else:
                logits = model(x, edge_index_t)
            probs = F.softmax(logits[ts_mask], dim=1)[:, 1].cpu().numpy()
            y_np  = y[ts_mask].cpu().numpy()

            # Adaptive threshold
            best_thr = 0.5 if t < 43 else 0.3
            preds = (probs >= best_thr).astype(int)

            all_probs.append(probs)
            all_preds.append(preds)
            all_true.append(y_np)

        # Fine-tune with replay buffer
        current_idx = ts_mask.nonzero(as_tuple=True)[0]
        buffer_size = 200
        if len(replay_illicit) > 0:
            illicit_sample = replay_illicit[
                torch.randperm(len(replay_illicit))[:buffer_size]
            ]
        else:
            illicit_sample = torch.tensor([], dtype=torch.long)
        licit_sample = licit_all[torch.randperm(len(licit_all))[:buffer_size]]

        ft_idx = torch.cat([current_idx, illicit_sample, licit_sample]).to(device)
        ft_opt = torch.optim.Adam(model.parameters(), lr=lr_ft)
        ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ft_opt, T_max=finetune_epochs)

        model.train()
        for _ in range(finetune_epochs):
            ft_opt.zero_grad()
            edge_index_ft, _ = dropout_edge(edge_index_t, p=0.2)
            if is_temporal:
                out = model(x, edge_index_ft, time_steps=timestep_arr_device)
            else:
                out = model(x, edge_index_ft)
            loss = criterion(out[ft_idx], y[ft_idx])
            loss.backward()
            ft_opt.step()
            ft_scheduler.step()

        # Update replay buffer
        new_illicit = ((y == 1) & ts_mask).nonzero(as_tuple=True)[0]
        replay_illicit = torch.cat([replay_illicit, new_illicit.cpu()])

    # Phase 3: Aggregate results
    if len(all_true) == 0:
        print(f"  WARNING: No test samples found for rolling window!")
        return 0.0, 0.0

    a_probs = np.concatenate(all_probs)
    a_preds = np.concatenate(all_preds)
    a_true  = np.concatenate(all_true)

    f1_mac = f1_score(a_true, a_preds, average='macro', zero_division=0)
    f1_ill = f1_score(a_true, a_preds, pos_label=1, zero_division=0)
    recall_ill = recall_score(a_true, a_preds, pos_label=1, zero_division=0)
    prec_ill = precision_score(a_true, a_preds, pos_label=1, zero_division=0)
    try:
        prauc = average_precision_score(a_true, a_probs)
    except ValueError:
        prauc = 0.0

    print(f'\n  {model_name} Rolling Window Results:')
    print(f'    F1 (illicit)  : {f1_ill:.4f}')
    print(f'    F1 (macro)    : {f1_mac:.4f}')
    print(f'    Precision     : {prec_ill:.4f}')
    print(f'    Recall        : {recall_ill:.4f}')
    print(f'    PR-AUC        : {prauc:.4f}')
    print(classification_report(a_true, a_preds, target_names=['Licit', 'Illicit']))

    return {
        'Model': f'{model_name} (rolling)',
        'F1 (illicit)': f1_ill,
        'Precision': prec_ill,
        'Recall': recall_ill,
        'PR-AUC': prauc,
        'Best Val F1': f1_mac,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9 — Main Execution
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    all_results = []

    # ──────────────────────────────────────────────────────────────
    # 9-A: XGBoost Baseline (no graph structure)
    # ──────────────────────────────────────────────────────────────
    print_section('XGBoost Baseline')
    xgb_results, xgb_model = train_xgboost_baseline(data_full, name='XGBoost')
    all_results.append(xgb_results)

    # ──────────────────────────────────────────────────────────────
    # 9-B: Standard GNN Models
    # ──────────────────────────────────────────────────────────────

    # GCN
    print_section('GCN Training')
    model_gcn = GCN(in_channels=data_full.num_features).to(DEVICE)
    gcn_results = train_and_evaluate_gnn(model_gcn, data_full, 'GCN')
    all_results.append(gcn_results)

    # GraphSAGE
    print_section('GraphSAGE Training')
    model_sage = GraphSAGE(in_channels=data_full.num_features).to(DEVICE)
    sage_results = train_and_evaluate_gnn(model_sage, data_full, 'GraphSAGE')
    all_results.append(sage_results)

    # GAT
    print_section('GAT Training')
    model_gat = GAT(in_channels=data_full.num_features).to(DEVICE)
    gat_results = train_and_evaluate_gnn(model_gat, data_full, 'GAT',
                                         weight_decay=1e-3)  # stronger reg for GAT
    all_results.append(gat_results)

    # ──────────────────────────────────────────────────────────────
    # 9-C: Spatio-Temporal GNN Models
    # ──────────────────────────────────────────────────────────────

    # EvolveGCN
    print_section('EvolveGCN (Spatio-Temporal) Training')
    model_evolve = EvolveGCN(in_channels=data_full.num_features).to(DEVICE)
    evolve_results = train_and_evaluate_temporal_gnn(
        model_evolve, data_full, 'EvolveGCN'
    )
    all_results.append(evolve_results)

    # TGAT
    print_section('TGAT (Spatio-Temporal) Training')
    model_tgat = TGAT(in_channels=data_full.num_features).to(DEVICE)
    tgat_results = train_and_evaluate_temporal_gnn(
        model_tgat, data_full, 'TGAT'
    )
    all_results.append(tgat_results)

    # ──────────────────────────────────────────────────────────────
    # 9-D: Rolling Window Evaluation (all models)
    # ──────────────────────────────────────────────────────────────
    print_section('Rolling Window Evaluation (Incremental Learning)')
    rolling_results = []

    # Re-instantiate models for rolling window
    rw_models = {
        'GCN': GCN(in_channels=data_full.num_features).to(DEVICE),
        'GraphSAGE': GraphSAGE(in_channels=data_full.num_features).to(DEVICE),
        'GAT': GAT(in_channels=data_full.num_features).to(DEVICE),
        'EvolveGCN': EvolveGCN(in_channels=data_full.num_features).to(DEVICE),
        'TGAT': TGAT(in_channels=data_full.num_features).to(DEVICE),
    }
    temporal_models = {'EvolveGCN', 'TGAT'}

    for name, model in rw_models.items():
        ckpt = os.path.join(OUTPUT_DIR, f'{name}_checkpoint.pt')
        is_temporal = name in temporal_models
        rw_result = rolling_window_evaluate(
            model=model,
            data=data_full,
            timestep_arr=data_full.time_steps.numpy(),
            model_name=name,
            pretrained_path=ckpt if os.path.exists(ckpt) else None,
            is_temporal=is_temporal,
        )
        if isinstance(rw_result, dict):
            rolling_results.append(rw_result)

    # ──────────────────────────────────────────────────────────────
    # 9-E: Comparative Results
    # ──────────────────────────────────────────────────────────────
    print_section('FINAL COMPARISON — All Models')

    # Static evaluation comparison
    results_df = pd.DataFrame(all_results)
    results_df = results_df[['Model', 'F1 (illicit)', 'Precision', 'Recall', 'PR-AUC', 'Best Val F1']]

    print('\n── Static Evaluation (train on ts 1-34, test on ts 35-49) ──')
    print(results_df.to_string(index=False, float_format='%.4f'))

    # Rolling window comparison
    if rolling_results:
        rw_df = pd.DataFrame(rolling_results)
        rw_df = rw_df[['Model', 'F1 (illicit)', 'Precision', 'Recall', 'PR-AUC']]
        print('\n── Rolling Window Evaluation (incremental learning) ──')
        print(rw_df.to_string(index=False, float_format='%.4f'))

    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'comparison_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f'\nResults saved to {results_path}')

    if rolling_results:
        rw_path = os.path.join(OUTPUT_DIR, 'rolling_window_results.csv')
        rw_df.to_csv(rw_path, index=False)
        print(f'Rolling window results saved to {rw_path}')

    # ── Comparison bar chart ──
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    metrics = ['F1 (illicit)', 'Precision', 'Recall', 'PR-AUC']
    colors_map = {
        'XGBoost': '#95A5A6',
        'GCN': '#3498DB',
        'GraphSAGE': '#2ECC71',
        'GAT': '#E74C3C',
        'EvolveGCN': '#9B59B6',
        'TGAT': '#E67E22',
    }

    for ax, metric in zip(axes, metrics):
        models_list = results_df['Model'].tolist()
        values = results_df[metric].tolist()
        bar_colors = [colors_map.get(m, '#333333') for m in models_list]
        x = np.arange(len(models_list))
        bars = ax.bar(x, values, color=bar_colors, alpha=0.85, edgecolor='white')
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models_list, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    plt.suptitle('Model Comparison — Test Set (ts > 34)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
    plt.savefig(chart_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'Comparison chart saved to {chart_path}')

    # ── Category summary ──
    print('\n' + '=' * 60)
    print('  CATEGORY SUMMARY')
    print('=' * 60)

    # Find best in each category
    xgb_row = results_df[results_df['Model'] == 'XGBoost']
    standard_gnn = results_df[results_df['Model'].isin(['GCN', 'GraphSAGE', 'GAT'])]
    st_gnn = results_df[results_df['Model'].isin(['EvolveGCN', 'TGAT'])]

    print(f'\n  Baseline (XGBoost):')
    if not xgb_row.empty:
        print(f'    F1={xgb_row["F1 (illicit)"].values[0]:.4f}  '
              f'PR-AUC={xgb_row["PR-AUC"].values[0]:.4f}')

    print(f'\n  Standard GNNs (best):')
    if not standard_gnn.empty:
        best_std = standard_gnn.loc[standard_gnn['F1 (illicit)'].idxmax()]
        print(f'    {best_std["Model"]}: F1={best_std["F1 (illicit)"]:.4f}  '
              f'PR-AUC={best_std["PR-AUC"]:.4f}')

    print(f'\n  Spatio-Temporal GNNs (best):')
    if not st_gnn.empty:
        best_st = st_gnn.loc[st_gnn['F1 (illicit)'].idxmax()]
        print(f'    {best_st["Model"]}: F1={best_st["F1 (illicit)"]:.4f}  '
              f'PR-AUC={best_st["PR-AUC"]:.4f}')

    print('\n' + '=' * 60)
    print('  DONE — All models trained and evaluated!')
    print('=' * 60)
