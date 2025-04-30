## End-to-End Pipeline Execution Guide

This guide outlines the complete end-to-end usage of the distributed LAION embedding and search pipeline.

1.  **Prepare the Environment**

    Navigate to your home directory:

    ```bash
    cd ~
    ```

    Move into the configuration directory and install all required Python dependencies:

    ```bash
    cd laion-distributed-pipeline/config
    python3 -m ensurepip --upgrade
    python3 -m pip install -r requirements.txt
    ```

2.  **Launch the Distributed Pipeline**

    Move into the pipeline orchestration directory:

    ```bash
    cd ../infra/ansible/pipeline
    ```

    Submit the SLURM-based distributed jobs in the following sequence:

    * **Run Distributed Embedding Jobs:**

        ```bash
        ansible-playbook -i ../../terraform/generate_inventory.py run-embedding-job.yaml
        ```

        > This will download the LAION dataset (parquet files) into `/home/almalinux/nfs/laion`.

    * **Run Distributed FAISS Indexing Jobs:**

        ```bash
        ansible-playbook -i ../../terraform/generate_inventory.py run-faiss-index-job.yaml
        ```

        > This step builds FAISS shards from the distributed CLIP embeddings.
        > Wait 5 - 10 minutes for this task to complete (test run for examination).

    * **Merge FAISS Index Shards into a Unified Index:**

        ```bash
        ansible-playbook -i ../../terraform/generate_inventory.py run-merge-faiss-job.yaml
        ```

        > After merging, a `faiss_index` directory will appear locally at:
        > `/home/almalinux/laion-distributed-pipeline/faiss_index`

3.  **Perform a Search Query**

    Move into the scripts directory:

    ```bash
    cd ../../../scripts
    ```

    Run the FAISS search script:

    ```bash
    python3 search_faiss_index.py
    ```

    Follow the on-screen prompts to select a query image.

    **Results will be:**

    > Printed directly to the terminal.
    >
    > Automatically saved to `/home/almalinux/laion-distributed-pipeline/search_outputs`.

**Notes:**

> * Ensure all distributed jobs complete successfully before proceeding to the next stage.
> * The search script requires the merged FAISS index (`merged.index`) and metadata (`merged_metadata.parquet`) to be present in the `faiss_index/` directory.