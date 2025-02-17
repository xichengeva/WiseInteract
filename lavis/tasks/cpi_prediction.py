import os
import torch
import json
import logging
import numpy as np
import pandas as pd
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.common.dist_utils import main_process
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, confusion_matrix, precision_score, recall_score, auc, accuracy_score, precision_recall_curve
from sklearn.metrics import accuracy_score

from sklearn import metrics

@registry.register_task("cpi_pretrain") # stage1 
class CPIPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True): # evaluation is here
        pass
        

@registry.register_task("protein_smiles_pretrain") # stage2
class ProteinSMILESPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass

@registry.register_task("finetune_cpi") # ft_cpi
class ProteinSMILESFinetuneTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass

def calculateMetric(targets, predictions):
    fpr, tpr, thresholds = roc_curve(targets, predictions)
    precision = tpr / (tpr + fpr)
    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    thred_optim = thresholds[5:][np.argmax(f1[5:])]
    y_pred_label = [1 if i else 0 for i in (predictions >= thred_optim)]

    Accuracy = accuracy_score(targets, y_pred_label)
    Precision = precision_score(targets, y_pred_label)
    Recall = recall_score(targets, y_pred_label)
    AUC = roc_auc_score(targets, predictions)
    tpr, fpr, _ = precision_recall_curve(targets, predictions)
    PRC = auc(fpr, tpr)
    F1 = f1_score(targets, y_pred_label)
    metrics = {"acc": Accuracy, "precision": Precision, "recall": Recall, "AUC": AUC, "prc":PRC, 'f1':F1}
    return metrics

def min_max_normalize(x):
    return (x - x.min()) / (x.max() - x.min())

@registry.register_task("EvaluateCPI")
class ProteinSMILESEvaluation(BaseTask):
    def __init__(self):
        super().__init__()
        self.epoch = 0

    def valid_step(self, model, samples):
        evaluation_way = self.cfg.run_cfg.evaluation_way
        test_path = self.cfg.run_cfg.test_path.split('/')[-1].split('.')[0]
        if evaluation_way == "pcm":
            predictions = model(samples, "pcm")
            # predictions = outputs.max(1)[1]
            return predictions.cpu().numpy() 
        elif evaluation_way == "pcc":
            sim = model(samples, "pcc").cpu().numpy()
            return torch.from_numpy(sim).cpu().numpy()
        elif evaluation_way == "pcc_map":
            sims = model(samples, "pcc_heatmap")
            return sims.cpu().numpy()
        elif evaluation_way == "pcc_pca":
            sims = model(samples, "pcc_pca", test_path)
            return sims.cpu().numpy()
        elif evaluation_way == "pcc_pcm_pertarget":
            sims, predictions = model(samples, "pcc_pcm", test_path)
            return sims.cpu().numpy(), predictions.cpu().numpy()
        elif evaluation_way == "pcc_pcm":
            sims, predictions = model(samples, "pcc_pcm", test_path)
            return sims.cpu().numpy(), predictions.cpu().numpy()

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        val_result = np.array(val_result).flatten()
        evaluation_way = self.cfg.run_cfg.evaluation_way
        datasets = self.cfg.run_cfg.datasets
        eval_result_file = '%s_%s.csv' % (datasets, evaluation_way) # _%s , self.epoch
        self.epoch = self.epoch + 1
        val_result = min_max_normalize(np.array(val_result))
        val_df = pd.DataFrame({
            'predictions': list(val_result)
        })
        val_df.to_csv(eval_result_file, index = False)
        metrics = self._report_metrics(
            eval_result_file=eval_result_file, split_name=split_name
        )
        return metrics


    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        predictions_df = pd.read_table(eval_result_file, sep=',')
        predictions = predictions_df['predictions']
        test_path = self.cfg.run_cfg.test_path ###
        data_val = pd.read_parquet(test_path) 
        targets = data_val['OUTCOME']

        metrics = calculateMetric(targets, predictions)
        log_stats = {split_name: {k: v for k, v in metrics.items()}}

        prefix = '/LAVIS/'
        evaluation_way = self.cfg.run_cfg.evaluation_way
        datasets = self.cfg.run_cfg.datasets
        with open(prefix + '%s_%s.txt' % (datasets, evaluation_way), "a") as f: ##
            f.write(test_path.split('/')[-1].split('.')[0] + '\t' + json.dumps(log_stats) + "\n")

        logging.info(metrics)
        return metrics

