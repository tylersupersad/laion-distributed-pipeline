# LAION Distributed Image Embedding + ANN Search

This repository implements a fully distributed CLIP embedding + FAISS ANN search pipeline for the LAION dataset using BeeGFS, SLURM, and a 4-node CPU cluster.

## Components

- **scripts/**: Embedding, indexing, search logic
- **infra/**: Infrastructure setup using Terraform + Ansible
- **beegfs🐝/**: Runtime data (not included in repo)

## How to Use

See [usage.md](usage.md) for full end-to-end instructions.

## Infrastructure

Provisioned via Terraform, configured using Ansible. See [cluster-config.md](cluster-config.md).