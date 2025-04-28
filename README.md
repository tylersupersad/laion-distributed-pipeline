## Distributed CLIP Embedding and Approximate Nearest Neighbor Search on LAION Dataset

This project implements a scalable, fully distributed pipeline for large-scale image-text data processing, utilizing the LAION2B-en-aesthetic dataset. The system performs high-throughput CLIP embedding generation and FAISS-based approximate nearest neighbor (ANN) search across a 5-node CPU cluster with NFS-shared storage and SLURM orchestration.

### System Overview

* **Distributed Embedding:** Parallel CLIP embedding of image batches across multiple worker nodes, with output persisted to NFS.
* **Parallel Indexing:** FAISS indices are constructed independently per partitioned embedding output, then merged into a unified search index.
* **Approximate Search:** The final FAISS index supports fast, scalable ANN retrieval on embedded representations.

### Architecture

**Cluster Configuration:**

* 1 Host node, 4 Worker nodes
* CPU-only inference with multi-node orchestration
* Centralized NFS storage for inputs, embeddings, and indices

**Orchestration:**

* SLURM array jobs for dynamic task distribution
* Terraform and Ansible automation for cluster provisioning and software setup

### Usage

This project requires the cluster to be configured before the pipeline can be executed. Please follow the steps outlined below:

1.  **Cluster Configuration ([cluster-config.md](cluster-config.md)):** This document provides a comprehensive guide to provisioning and configuring your 5-node CPU cluster, including network setup, storage access, and orchestration tools. **Start here.**
2.  **Pipeline Execution ([usage.md](usage.md)):** After successfully configuring the cluster, this document details how to analyze and run the distributed CLIP embedding and approximate nearest neighbor search pipeline on the LAION dataset.

### Key Features

* High concurrency support for embedding and indexing
* Monitoring via Prometheus and Node Exporter