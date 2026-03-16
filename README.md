# Graph Neural Networks for Bitcoin Transaction Classification

This project applies advanced Graph Neural Network (GNN) architectures to detect illicit Bitcoin transactions using the Elliptic dataset. The goal is to classify Bitcoin transactions as either **illicit** or **licit** by leveraging the temporal and structural properties of the Bitcoin transaction network.

## Project Overview

Bitcoin transactions form a complex network where nodes represent transactions and edges represent flows of value between them. This project exploits this graph structure to build predictive models that can identify fraudulent or illicit transactions with high accuracy.

### Key Features

- **Multiple GNN Architectures**: Implements and compares GCN (Graph Convolutional Network), GraphSAGE, and GAT (Graph Attention Network)
- **Research-Based Optimizations**: Incorporates normalization techniques (GraphNorm) and initialization strategies based on academic research
- **Comprehensive Analysis**: Includes data exploration, feature analysis, model comparison, and visualization
- **Production-Ready Metrics**: Evaluates models using F1-score, precision, recall, ROC-AUC, and average precision

## Dataset

### Elliptic Bitcoin Dataset

The project uses the **Elliptic dataset**, a real-world Bitcoin transaction dataset containing:

- **306,499 Bitcoin transactions** as nodes
- **1.76M transactions connections** as edges (representing value flows)
- **165 transaction features** including temporal and structural properties
- **Labels**: Transaction classification (Illicit, Licit, Unknown)

#### Data Files
```
elliptic_bitcoin_dataset/
├── elliptic_txs_features.csv      # 306,499 × 165 features per transaction
├── elliptic_txs_edgelist.csv      # Transaction connections/flows
└── elliptic_txs_classes.csv       # Transaction labels
```

### Data Characteristics

- **Temporal Structure**: Transactions distributed across 49 timesteps
- **Class Distribution**: Imbalanced dataset with majority licit transactions
- **Feature Engineering**: 165 features capturing transaction amounts, patterns, and temporal behavior

## Models

### GCN (Graph Convolutional Network)
- 2-layer baseline architecture with no normalization
- Simple yet effective for baseline performance comparison
- Dropout: 0.3, Hidden dim: 128

### GraphSAGE (Graph Sample and Aggregate)
- Uses mean aggregation over sampled neighborhoods
- Xavier initialization on classifier head
- Multi-layer architecture with configurable hidden dimensions
- Optimal configuration: 140 hidden dim, 103 embedding dim

### GAT (Graph Attention Network)
- Multi-head attention mechanism for learning adaptive aggregation weights
- GraphNorm normalization after each hidden layer
- Xavier initialization on classifier head
- Optimal configuration: 148 hidden dim, 89 embedding dim, 4 attention heads

## Performance Metrics

Models are evaluated using:
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **ROC-AUC**: Area under receiver operating characteristic curve
- **Average Precision**: Weighted average of precision at different thresholds
- **Confusion Matrix**: Detailed breakdown of classification results

## Key Findings

The project demonstrates that:
1. **Graph structure matters**: GNNs significantly outperform traditional ML by leveraging transaction relationships
2. **Normalization improves stability**: GraphNorm in GAT provides more robust training
3. **Attention helps generalization**: GAT with proper initialization shows best generalization
4. **Temporal patterns are informative**: Transaction timestep features enhance classification

## Requirements

### Python Libraries
```
numpy          # Numerical computing
pandas         # Data processing
matplotlib     # Basic visualization
seaborn        # Statistical visualization
networkx       # Graph analysis
torch           # Deep learning framework
torch_geometric # Graph neural network library
scikit-learn   # ML metrics and preprocessing
```

### Installation

```bash
pip install torch torch_geometric scikit-learn matplotlib seaborn networkx pandas numpy
```

## Project Structure

```
gnn/
├── main.ipynb                           # Main Jupyter notebook with full analysis
└── elliptic_bitcoin_dataset/
    ├── elliptic_txs_features.csv       # Transaction features
    ├── elliptic_txs_edgelist.csv       # Transaction graph edges
    └── elliptic_txs_classes.csv        # Transaction labels
```

## Notebook Contents

The `main.ipynb` notebook contains:

1. **Data Loading & Exploration**: Load features, edges, and labels
2. **Data Preprocessing**: Feature scaling, edge indexing, train/test splitting
3. **Model Architecture Definitions**: GCN, GraphSAGE, and GAT implementations
4. **Baseline Training**: Train GCN baseline model
5. **Hyperparameter Tuning**: Grid search for optimal model configurations
6. **Model Comparison**: Side-by-side evaluation of all three architectures
7. **Visualization**: Network statistics, performance curves, confusion matrices
8. **Analysis & Insights**: Detailed performance breakdown and conclusions

## Usage

### Running the Notebook

```bash
jupyter notebook main.ipynb
```

Then run all cells sequentially or run specific sections for detailed analysis.

### Training a Model

Models are trained with:
- **Optimizer**: Adam
- **Loss Function**: Cross-entropy loss
- **Early Stopping**: Based on validation loss
- **Device**: GPU (if available) or CPU

Training configuration can be adjusted in the hyperparameter tuning sections.

## Results & Benchmarks

Expected performance on the test set:
- **GCN**: F1-score ~0.65-0.70 (baseline)
- **GraphSAGE**: F1-score ~0.70-0.75 (improvement over baseline)
- **GAT**: F1-score ~0.72-0.77 (best performance)

*Note: Exact scores depend on train/test split and random seed*
