from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.cpi_datasets import CPIDataset
from lavis.common.registry import registry

@registry.register_builder("cpi_prediction")
class CPIBuilder(BaseDatasetBuilder):
    dataset_cls = CPIDataset
    DATASET_CONFIG_DICT = {
            "default": "configs/default_dataset.yaml"
        }

    def build_processors(self):
        proteins_proc_cfg = self.config.get("proteins_processor")
        smiles_proc_cfg = self.config.get("smiles_processor")

        if proteins_proc_cfg is not None: # vis is proteins
            proteins_train_cfg = proteins_proc_cfg.get("train")
            proteins_eval_cfg = proteins_proc_cfg.get("val")
            proteins_test_cfg = proteins_proc_cfg.get("test")

            self.vis_processors["train"] = self._build_proc_from_cfg(proteins_train_cfg)
            self.vis_processors["val"] = self._build_proc_from_cfg(proteins_eval_cfg)
            self.vis_processors["test"] = self._build_proc_from_cfg(proteins_test_cfg)

        if smiles_proc_cfg is not None: # txt is smiles
            smiles_train_cfg = smiles_proc_cfg.get("train")
            smiles_eval_cfg = smiles_proc_cfg.get("val")
            smiles_test_cfg = smiles_proc_cfg.get("test")

            self.text_processors["train"] = self._build_proc_from_cfg(smiles_train_cfg)
            self.text_processors["val"] = self._build_proc_from_cfg(smiles_eval_cfg)
            self.text_processors["test"] = self._build_proc_from_cfg(smiles_test_cfg)

        
        kw_proc_cfg = self.config.get("kw_processor")
        if kw_proc_cfg is not None:
            for name, cfg in kw_proc_cfg.items():
                self.kw_processors[name] = self._build_proc_from_cfg(cfg)
    
    def _download_ann(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        train_dataset = self.dataset_cls(
            root = build_info.train.url,
            protein_processor=self.vis_processors["train"],
            smiles_processor=self.text_processors["train"],
            datatype = build_info.use_neg,
        )
        val_dataset = self.dataset_cls(
            root = build_info.val.url,
            protein_processor=self.vis_processors["train"],
            smiles_processor=self.text_processors["train"],
            datatype = build_info.use_neg,
        )
        test_dataset = self.dataset_cls(
            root = build_info.test.url,
            protein_processor=self.vis_processors["train"],
            smiles_processor=self.text_processors["train"],
            datatype = build_info.use_neg,
        )
        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
            }
    
    def build_datasets(self):
        print('building datasets...')
        datasets = self.build()
        return datasets
