# FedFAW

<!-- TODO: add the venue once the paper is published -->
Official implementation of **Fisher Aggregation with Adaptive Client Weighting for One-Shot Federated Learning** (FedFAW).

## Overview

One-shot federated learning (OFL) trains collaboratively with a single round of communication, so the global model must come from a single aggregation step. Under non-IID client data, weighting clients by sample size does not reflect how useful each client model is, and applying one weight to all of a client's parameters ignores parameter-level importance.

FedFAW (**Fed**erated **F**isher **A**ggregation with Adaptive Client **W**eighting) combines parameter-level Fisher weighting with validation-guided client weights:

1. **Local training.** Each client *i* trains a local model θ_i from a shared initialization and computes the diagonal empirical Fisher f_i (squared per-sample gradients of the log-likelihood, averaged over its data). It uploads both once.
2. **Fisher-weighted candidate aggregate.** For client weights α on the probability simplex, the server forms, element-wise,

   ```
   θ_agg(α) = Σ_i α_i f_i ⊙ θ_i  /  ( Σ_i α_i f_i + ε )
   ```

3. **Bayesian optimization of α.** The server searches raw variables a ∈ [0, 1]^K, maps them to the simplex by α = a / ‖a‖₁, and scores each candidate aggregate by its accuracy on a small server-side validation set. It uses a Gaussian-process surrogate with the UCB acquisition function (κ = 5), 30 initial points and 75 optimization iterations. The best α* gives the global model.

Across FMNIST, CIFAR-10, CINIC-10 and SVHN at three heterogeneity levels, FedFAW achieves the highest mean accuracy in all 12 settings, exceeding the strongest baseline by 3.3 percentage points on average and by up to 9.7. On CIFAR-10 it uploads about 60% less than FedFisher-KFAC and FedLPA and 86.7% less than FedBEns.

## Installation

Tested on Linux with Python 3.9 and CUDA 11.8.

```bash
git clone https://github.com/nclabteam/FedFAW.git
cd FedFAW
conda env create -f environment.yml
conda activate fedfaw
```

`environment.yml` installs Python 3.9 and everything in `requirements.txt` (PyTorch 2.7.1 with CUDA 11.8 wheels). Without conda, use `pip install -r requirements.txt` in a Python 3.9 environment. `environment.lock.yml` is an exact snapshot of the environment used for the experiments.

### Datasets

| `--dataset` | Dataset | Preparation |
|---|---|---|
| `fashion_mnist` | Fashion-MNIST | downloaded automatically to `./datasets/` |
| `cifar10` | CIFAR-10 | downloaded automatically |
| `svhn` | SVHN | downloaded automatically |
| `cinic10` | CINIC-10 | download by hand and extract to `./datasets/cinic-10/` (with `train/` and `test/`) |

### Pretrained checkpoint

`resnet_18_tiny_imagenet_40.pt` holds the pretrained ResNet-18 weights used by the pretrained-backbone experiments (`--model ResNet18 --pretrained True`; path set by `common.pretrained_path`).

## Running

```bash
# FedFAW on CIFAR-10, Dirichlet β = 0.1, 5 clients, seed 0, GPU 0
python main.py --config configs/base.yaml --strategy fedfaw --fisher_type diagonal \
               --dataset cifar10 --alpha 0.1 --seed 0 --gpu 0

# a baseline
python main.py --config configs/base.yaml --strategy fedfisher --fisher_type kfac \
               --dataset cifar10 --alpha 0.1 --seed 0 --gpu 0
```

`configs/base.yaml` holds the main one-shot setting (1 round, 5 clients, 32 local epochs, SGD with lr 0.01 and momentum 0.9, the paper's CNN `FedNet`). `configs/base_svhn.yaml` is the same setting for SVHN. Command-line flags override the config:

| Flag | Meaning |
|---|---|
| `--strategy` | method (table below) |
| `--fisher_type` | `diagonal` or `kfac`, for the Fisher-based methods |
| `--dataset` | `fashion_mnist`, `cifar10`, `cinic10`, `svhn` |
| `--model` | `FedNet` (default) or `ResNet18` |
| `--pretrained True` | start ResNet-18 from the pretrained checkpoint |
| `--alpha` | Dirichlet concentration β of the client split |
| `--num_clients`, `--epochs` | number of clients K, local epochs |
| `--num_val_samples` | size of the server-side validation set (default 500) |
| `--init_points`, `--n_iter`, `--acq`, `--kappa` | FedFAW Bayesian-optimization settings |
| `--seed`, `--gpu`, `--base_dir` | random seed, GPU index, output root |

### Methods

| Method in the paper | Command |
|---|---|
| **FedFAW** (ours) | `--strategy fedfaw --fisher_type diagonal` |
| FedFAW w/o Fisher (ablation) | `--strategy fedfaw_nofisher` |
| FedAvg | `--strategy fedavg` |
| OTFusion | `--strategy otfusion` |
| RegMean | `--strategy regmean` |
| DENSE | `--strategy dense` |
| FisherMerge | `--strategy fisher_merge --fisher_type diagonal` |
| FedLPA | `--strategy fedlpa` |
| FedFisher-Diagonal | `--strategy fedfisher --fisher_type diagonal` |
| FedFisher-KFAC | `--strategy fedfisher --fisher_type kfac` |
| FedBEns | `--strategy fedbens` |

Method-specific hyperparameters are in the matching block of `configs/base.yaml` (`fedfaw`, `otfusion`, `regmean`, `dense`, `fedlpa`, `fedbens`).

### Server-side validation set

The validation set (500 samples by default) is drawn uniformly at random, without replacement, from the dataset's official **test** split. It is the same for every seed. Those samples are removed from the test set, so all reported accuracies are computed on the remaining test samples, and validation and test data never overlap. FedFAW and FedBEns both use this set.

## Reproducing the experiments

All results are mean ± standard deviation over five seeds (0, 13, 44, 123, 456).

| Experiment | Settings |
|---|---|
| Statistical heterogeneity (main table) | all four datasets, `--alpha` ∈ {0.05, 0.1, 0.3}, every method |
| Component ablation | CIFAR-10, `--alpha` ∈ {0.05, 0.1, 0.3}: `fedavg`, `fedfisher --fisher_type diagonal`, `fedfaw_nofisher`, `fedfaw` |
| Scalability | CIFAR-10, `--alpha 0.1`, `--num_clients` ∈ {5, 10, 20} |
| Pretrained backbone | CIFAR-10, `--model ResNet18 --pretrained True`, client `lr: 0.001`, `--alpha` ∈ {0.05, 0.1, 0.3} |
| Few-round FL | FMNIST, CIFAR-10, CINIC-10, `--alpha 0.1`, `server.num_rounds: 5`, `--epochs 10` |
| Validation-set size | CIFAR-10, `--alpha 0.1`, `--num_val_samples` ∈ {50, …, 1000}, `fedfaw` and `fedbens` |
| Validation distribution shift | CIFAR-10, `--alpha 0.1`, `configs/val_data_shift.yaml` with `val_data_shift.shift_type` = `label` (`label_shift_dir_alpha`), `covariate` (brightness, `covariate_shift_strength`) or `domain` (`domain_shift_dataset: cinic10`) |

`scripts/run_experiments.sh` and `scripts/run_experiments_multijob.sh` queue every (dataset × β × method × seed) combination across the listed GPUs. The lists are set at the top of each script.

```bash
bash scripts/run_experiments.sh                          # one job per GPU
JOBS_PER_GPU=3 bash scripts/run_experiments_multijob.sh  # several jobs per GPU
```

## Outputs

Each run writes to `<base_dir>/<date>/<strategy>/<run index>/`:

| File | Content |
|---|---|
| `config.json` | full configuration of the run |
| `<strategy>_<dataset>_<batch>_<lr>_<epochs>.csv` | per round: test accuracy, test loss, time, upload and download MiB |
| `<strategy>_<dataset>_<batch>_<lr>_<epochs>.log` | training log |
| `bo_summary.json` | FedFAW only: α*, its validation accuracy, every Bayesian-optimization evaluation and the server time |

Summaries over seeds:

```bash
python scripts/analyze_results.py       --paths out/   # mean ± std per method, dataset and β
python scripts/analyze_results_table.py --paths out/   # dataset × method table
python scripts/analyze_multi_round.py   --paths out/   # few-round results
python scripts/plot_convergence.py      --paths out/ --datasets cifar10 --alpha 0.1
```

## Project structure

```
main.py                         entry point: python main.py --config <config.yaml> [flags]
configs/                        experiment configurations
resnet_18_tiny_imagenet_40.pt   pretrained ResNet-18 weights
mak/
  models.py                     FedNet (paper CNN), ResNet-18
  clients/                      local training; Fisher computation (fisher_client.py)
  servers/                      one server per method, including fedfaw_server.py
  utils/
    fedfaw_utils.py             Fisher-weighted aggregate and Bayesian optimization of α
    data_fedfisher.py           dataset loading and Dirichlet partitioning
    data_shift.py               server-side validation set and validation distribution shift
    ...                         per-method utilities and communication accounting
scripts/                        experiment launchers and result analysis
```

<!-- ## Citation -->

<!-- TODO: complete the entry once the paper is published
If you use this code, please cite:

```bibtex
@article{kundroo2026fedfaw,
  title  = {Fisher Aggregation with Adaptive Client Weighting for One-Shot Federated Learning},
  author = {Kundroo, Majid and Kim, Taehong},
  year   = {2026}
}
``` -->

## Acknowledgments

<!-- TODO: funding acknowledgment -->

The baselines follow their official implementations, including [FedBEns](https://github.com/jacopot96/FedBEns).

## License

This project is released under the [MIT License](LICENSE).
