LapRLS-GAT: A Two-Stage Framework for Interaction Prediction
Overview

This repository implements a two-stage prediction framework for microbial interaction prediction (e.g., Lb–St strain interactions).
The framework integrates Laplacian Regularized Least Squares (LapRLS) with a Graph Attention Network (GAT) to effectively leverage similarity information, soft labels, and graph-structured data.

Stage 1 (LapRLS):
Multi-source similarity matrices are fused to generate soft interaction scores for all candidate strain pairs.

Stage 2 (LapRLS-GAT):
The LapRLS soft predictions are incorporated as edge attributes to construct graphs, which are then fed into a GAT model for final prediction using 5-fold cross-validation.

