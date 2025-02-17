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

@registry.register_task("EvaluateGeneration")
class SMILESEGenerationEvaluation(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__()
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate
        report_metric = run_cfg.get("report_metric", True)
        return cls(num_beams=num_beams, max_len=max_len, min_len=min_len, evaluate=evaluate, report_metric=report_metric)

    def valid_step(self, model, samples):
        # molList = []
        # seqIndexList = []
        for i in range(10):
            # mols = model.predict_answers(
            #     samples,
            #     use_nucleus_sampling=False,
            #     num_beams=self.num_beams,
            #     max_length=self.max_len,
            #     min_length=self.min_len,
            #     prompt="CC"
            # )
            mols = model.generate(
                samples,
                use_nucleus_sampling=False,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
            )
            # seqIndexes = samples["seqIndex"]
            # for mol, seqIndex in zip(mols, seqIndexes):
            #     molList.append(mol)
            #     seqIndexList.append(seqIndex)
            df = pd.DataFrame({'smiles': mols}) #, 'seqIndex':seqIndexList
            df.to_csv('test_pi3k_4_10.csv', index = False, mode='a', header=None) 
        return mols
    