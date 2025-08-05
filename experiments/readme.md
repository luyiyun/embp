# EM-based Biomarker Effect Estimate for Combining Data from Multiple Biomarker Studies

## Reproduce the results

To reproduce the results, you can run the following code:

1. Clone the repository:

    ```bash
    git clone https://github.com/luyiyun/bayesian_biomark_pooling.git
    cd bayesian_biomark_pooling
    git checkout embp
    ```

2. Create a new conda/mamba environment and install the required packages:

    ```bash
    mamba create -n bbp_env python=3.11 -y
    mamba activate bbp_env
    pip install -e ".[develop]"  # install all dependencies for development in editable mode
    ```

3. Activate the conda environment:

    ```bash
    mamba activate bbp_env
    ```

4. Run the commands in `experiments/embp/main.sh` to reproduce the results:

    ```bash
    cd experiments/embp
    bash main.sh
    ```

5. (Optional) If you want to generate the customized simulation data and run the model on them, you can run the following command:

    ```bash
    cd experiments/embp
    python main.py simulate --n_studies 4 --n_samples 20 --output_dir data/simu
    python main.py analyze --data_dir data/simu --output_dir results/simu
    python main.py evaluate --analyzed_dir results/simu --output_file evaluation.csv
    ```
    you can run `python main.py --help` to see all available commands and options.
