# System Architecture

## Overview
The Traffic Forecasting System is designed as a modular pipeline:
1.  **Data Ingestion**: Raw CSV/H5 files -> `TrafficDataset`
2.  **Graph Construction**: Sensor locations -> Adjacency Matrix -> PyG `Data` object
3.  **Model**: `ST-GNN` (Temporal Encoder -> Graph Layer -> Temporal Decoder)
4.  **Inference**: FastAPI server serving predictions via REST API

## Components
-   `src.models.st_gnn`: Main model assembly.
-   `src.models.temporal_encoder`: GRU-based history encoder.
-   `src.models.graph_layers`: GATv2Conv layers for spatial message passing.
-   `src.models.temporal_decoder`: GRU-based future predictor.

## Data Flow
`[Batch, Time_In, Nodes, Feat]` -> **Encoder** -> `[Batch, Nodes, Hidden]` -> **Graph Layer** -> `[Batch, Nodes, Hidden]` -> **Decoder** -> `[Batch, Time_Out, Nodes, 1]`
