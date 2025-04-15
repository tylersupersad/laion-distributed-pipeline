# LAION Distributed Embedding Cluster Configuration

This document details the configuration and setup of a distributed cluster designed for large-scale CLIP image embedding using the LAION Aesthetic dataset. The system leverages NFS for distributed storage, SLURM for efficient job scheduling, and Ansible for comprehensive cluster automation.

---

## Cluster Architecture

| Role          | Hostname      | Resources         | Purpose                                |
|---------------|---------------|-------------------|----------------------------------------|
| Head Node     | `hostnode`    | 2 CPUs, 4GB RAM   | SLURM controller, job submission       |
| Worker Node 1 | `workernode1` | 4 CPUs, 32GB RAM  | Image embedding processing             |
| Worker Node 2 | `workernode2` | 4 CPUs, 32GB RAM  | Image embedding processing             |
| Worker Node 3 | `workernode3` | 4 CPUs, 32GB RAM  | Image embedding processing             |
| Worker Node 4 | `workernode4` | 4 CPUs, 32GB RAM  | Image embedding processing             |

> **Key Features:**
>
> * All nodes utilize shared storage via NFS client.
> * Passwordless SSH access is configured across all nodes for seamless operation.
> * Cluster deployment and configuration are fully automated using Ansible.

---

## Repository Setup and Cluster Deployment

1.  **Clone the Repository and Navigate to the Terraform Directory:**

    ```bash
    git clone [https://github.com/tylersupersad/laion-distributed-pipeline.git](https://github.com/tylersupersad/laion-distributed-pipeline.git)
    cd laion-distributed-pipeline/infra/terraform
    ```

2.  **Configure Infrastructure Variables:**

    * Edit `variables.tf` to customize infrastructure settings such as region, instance types, and other provider-specific parameters.

3.  **Provision the Cluster using Terraform:**

    ```bash
    terraform init
    terraform apply
    ```

    > This step will provision the virtual machines and network resources based on the configuration.

4.  **Cluster Bootstrapping with Ansible:**

    * **Prepare the Environment:**

        ```bash
        chmod +x ../generate_inventory.py
        export HF_TOKEN=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
        ```

    * **Run the Ansible Playbook:**

        ```bash
        cd ../ansible/setup
        ansible-playbook -i ../../terraform/generate_inventory.py full.yaml
        ```

    * **Ansible Automation Tasks:**

        * Install necessary Python packages and system dependencies.
        * Configure BeeGFS or NFS client based on node roles.
        * Set up SLURM for job scheduling (controller and worker nodes).
        * Prepare the environment for CLIP image embedding processing.

---