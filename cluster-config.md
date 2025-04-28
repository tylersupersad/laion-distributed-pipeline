# LAION Distributed Embedding Cluster Configuration

This document details the configuration and setup of a distributed cluster designed for large-scale CLIP image embedding using the LAION Aesthetic dataset. The system leverages NFS for distributed storage, SLURM for efficient job scheduling, and Ansible for comprehensive cluster automation.

---

## Cluster Architecture

| Role          | Hostname      | Resources         | IP Address    |
|---------------|---------------|-------------------|---------------|
| Head Node     | `hostnode`    | 2 CPUs, 4GB RAM   | 10.134.12.239 |
| Worker Node 1 | `workernode1` | 4 CPUs, 32GB RAM  | 10.134.12.245 |
| Worker Node 2 | `workernode2` | 4 CPUs, 32GB RAM  | 10.134.12.240 |
| Worker Node 3 | `workernode3` | 4 CPUs, 32GB RAM  | 10.134.12.250 |
| Worker Node 4 | `workernode4` | 4 CPUs, 32GB RAM  | 10.134.12.243 |

> **Note:** The cluster deployment and configuration described above were initiated from the machine with IP address `10.134.12.252`.
>
> **Key Features:**
>
> * All nodes utilize shared storage via NFS client.
> * Passwordless SSH access is configured across all nodes for seamless operation.
> * Cluster deployment and configuration are fully automated using Ansible.

---

## Repository Setup and Cluster Deployment

1.  **Prepare Environment:**
    * Install Git (if not already installed):
    ```bash
    sudo dnf install git
    ```
    * Clone the repository:
    ```bash
    git clone https://github.com/tylersupersad/laion-distributed-pipeline.git
    ```
    * Navigate to the `terraform` directory to initiate the provisioning of the cluster: 
    ```bash
    cd laion-distributed-pipeline/infra/terraform
    ```

2.  **Configure Infrastructure Variables:**

    * Edit `variables.tf` to customize infrastructure settings such as region, instance types, and other provider-specific parameters.

3.  **Provision the Cluster using Terraform:**

    ```bash
    terraform init
    terraform apply
    ```

    **Notes:**

    > - Follow the prompts and type `yes` when asked for confirmation.
    > - This will automatically create all required virtual machines and network resources based on the Terraform files.

4.  **Cluster Bootstrapping with Ansible:**

    * **Ensure generate_inventory.py is Executable:**
        ```bash
        chmod +x generate_inventory.py
        ```

    * **Navigate to the `setup` Directory:**
        ```bash
        cd ../ansible/setup
        ```

    * **Prepare the Environment (HuggingFace Requires Token for Dataset Download):**
        ```bash
        export HF_TOKEN=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
        ```

    * **Run the Ansible Playbook:**
        ```bash
        ansible-playbook -i ../../terraform/generate_inventory.py full.yaml
        ```

5.  **Configure Prometheus Monitoring:**

    * **Navigate to the `monitoring` Directory:**
        ```bash
        cd ../monitoring
        ```

    * **Deploy Prometheus and Node Exporter with Ansible:**
        ```bash
        ansible-playbook -i ../../terraform/generate_inventory.py install_monitoring.yaml
        ```

    * **Port Forward Prometheus UI (example with Hostnode IP 10.134.12.239):**
        ```bash
        ssh -i %USERPROFILE%\.ssh\condenser -J condenser-proxy -L 9090:localhost:9090 almalinux@10.134.12.239
        ```

    * **Access Prometheus UI:**
        Open http://localhost:9090 in your browser.

* **Ansible Automation Tasks:**

    * Install necessary Python packages and system dependencies.
    * Configure NFS client based on node roles.
    * Set up SLURM for job scheduling (controller and worker nodes).
    * Prepare the environment for CLIP image embedding processing.
    * Deploy Prometheus and Node Exporter for distributed system monitoring.

---