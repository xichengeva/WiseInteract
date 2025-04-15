# Highly efficient discovery of active compounds against protein sequences with WiseInteract
WiseInteract, a state-of-the-art method for compound protein interaction prediction. This repository contains all code, instructions, and model weights necessary to run the method or to retrain a model. If you have any questions, feel free to open an issue or reach out to us: niubuying@simm.ac.cn, xicheng@simm.ac.cn

# Description

**WiseInteract** is an open-source method for sequence-based compound protein interaction prediction. 

**Things WiseInteract can do**
- Virtual screening
- Target discovering

# Setup Environment
## 1. Clone the current repo
```bash
git clone https://github.com/niubuying/WiseInteract
```
## 2.Installation
```bash
conda create -n cpi python=3.8
conda activate cpi
conda install -c conda-forge rdkit
pip install dgl
pip install dgllife
pip install salesforce-lavis==1.0.0
pip install --upgrade transformers==4.27
pip install fair-esm
pip install seaborn
```

# Running WiseInteract
## 1.Extract protein embeddings:
```bash
python protein.py
esm-extract esm2_t33_650M_UR50D protein.txt proteins_emb_esm2 --repr_layers 33 --include per_tok
```
## 2.Train:
```bash
python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path train.yaml
```
## 3.Predict:
```bash
python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path prediction.yaml
```
## 4.Related data and model are saved below:
[https://zenodo.org/records/14375583](https://zenodo.org/records/15220346)
