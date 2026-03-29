# Resource Utilization Analysis and Prediction of VASP GPU Jobs Running on Perlmutter
This repository contains code and notebooks for analyzing and predicting GPU resource utilization on the Perlmutter supercomputer. The analyses focus primarily on VASP (Vienna Ab initio Simulation Package) jobs, along with data from several additional GPU applications collected in March 2025.

**Author & Maintainer:** Beste Oztop, Boston University  
**Contact:** boztop@bu.edu

This research was supported by the National Energy Research Scientific Computing Center (NERSC), a U.S. Department of Energy Office of Science User Facility operated under Contract No. DE-AC02-05CH11231.

## Overview
This repository contains scripts and analysis workflows for characterizing and predicting GPU resource utilization and average power consumption of GPU jobs on NERSC’s Perlmutter system. The analyses are based on two system-level data sources: (1) Slurm accounting logs and (2) NVIDIA DCGM GPU monitoring metrics.


The goals of this repository are to:
1) Characterize the GPU utilization, GPU memory utilization, and average power usage of VASP jobs on Perlmutter, and demonstrate how the same analysis methodology generalizes to other production GPU applications.
2) Presenting a practical two-stage prediction framework that predicts resource usage before queueing jobs and predicts runtime power consumption during job execution, with direct relevance to scheduling and power-aware system management.
3) Support reproducibility and reuse by providing data analysis and ML model training scripts that can be adapted by system operators, middleware developers, and application users on other GPU-accelerated HPC platforms.


## Dataset Characteristics
We limit our analysis to GPU jobs that ran on regular and premium GPU nodes (i.e., there is no node sharing between jobs included in our analyses) on Perlmutter from March 2025. Using our analysis and prediction methods, one can extend our findings to other GPU applications and workloads.

Our input dataset includes:
- Accounting job data for GPU jobs from Slurm, and
- GPU utilization metrics from NVIDIA Data Center Management Tool (DCGM).

Due to data access restrictions and user privacy considerations, the raw datasets are not included or shared in this repository. The provided scripts and workflows are designed to operate on equivalent Slurm and DCGM data collected from other GPU-accelerated HPC systems.


| Slurm Submission Feature | Description                                   |
| ----------- | ---------------------------------------------------------- |
| `User`      | Name of the user submitting the job                        |
| `JobName`   | User-defined job name                                      |
| `Account`   | Allocation or account charged for the job                  |
| `Category`  | Scientific or application category associated with the job |
| `ReqCPUs`   | Number of CPUs requested                                   |
| `ReqNode`   | Number of GPU nodes requested                              |
| `ReqGPUs`   | Number of GPUs requested                                   |
| `ReqMem`    | Requested memory size                                      |
| `Timelimit` | Hard wall-clock time limit for the job                     |

| DCGM Metric  | Description                                                        |
| ----------------- | ------------------------------------------------------------------ |
| `gpu_utilization` | GPU utilization activity (%)                                       |
| `fb_used`         | Used GPU frame buffer memory (MB)                                  |
| `fb_free`         | Free GPU frame buffer memory (MB)                                  |
| `sm_active`       | Fraction of cycles where at least one warp is active on an SM      |
| `sm_occupancy`    | Ratio of active warps on an SM relative to the theoretical maximum |
| `dram_active`     | Fraction of cycles with active device memory transactions          |
| `fp64_active`     | Fraction of cycles where the FP64 pipeline is active               |
| `tensor_active`   | Fraction of cycles where any tensor pipeline is active             |
| `power_usage`     | Average GPU power consumption (W)                                  |



| Application Name | # of Jobs from Slurm| # of DCGM Data Points in Training Set |# of DCGM Data Points in Test Set|
|----------|----------|----------|----------|
| VASP  | 32,322  | 1625121  | 349086
| LAMMPS  | 134  | 20400  | 4368
| Espresso  | 363  | 49296  | 12685
| Atlas  | 655  | 116048  | 25890
| E3SM  | 267  | 8945 | 2257


## Repository Organization
The repository is organized into the following directories:
- `notebooks/`: Contains the following Jupyter notebooks:
  - `resource_util_analysis.ipynb`
    - Includes GPU and GPU memory utilization analysis, power consumption analysis, AI Performance Metrics and Roofline Analysis for VASP and other applications.
  - `stage_one_param_sweep.ipynb`
    - Includes hyperparameter tuning experiments for 4 different ML models.
  - `stage_one_pred_res.ipynb`
    - Includes experimental results for the **first stage, before job submission** predictions.
  - `stage_two_pred_res.ipynb`
    - Includes experimental results for the **second stage, during job submission** predictions, including confusion matrices and feature importances.
- `scripts/`: Contains the Python script for preparing the multivariate time series data for model training and the script for **during job execution** power prediction using real-time GPU utilization metrics.


## Reproducibility: Software Requirements
For the reader's reference, the analysis and prediction workflows in this repository require the Python libraries and software included in `requirements.txt`. The best practice is to create a virtual environment and load dependencies, simply as follows:

* Create a virtual environment (Python 3.6.15 used in our experiments)
```bash
python3 -m venv venv
```
* Activate the environment
```bash
source venv/bin/activate
```

* Upgrade pip
```bash
pip install --upgrade pip
```

* sInstall required packages
```bash
pip install -r requirements.txt
```

