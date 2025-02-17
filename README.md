# WiseInteract

## 1.Installation
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

## 2.train:
```bash
python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path pretrain_stage1.yaml
```
## 3.evaluate:
```bash
python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path prediction.yaml
```
## 4.related data and model are saved below:
https://zenodo.org/records/14375583
