import os

import torch


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq

from module.data import IndicHeadlineGenerationData
from module.metrics import metricDic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class trainer:
    def __init__(self, args):
        for key, value in args.items():
            setattr(self, key, value)
        
        try:
            os.makedirs(f"{self.log_dir}{self.run_name}")
        except:
            print(f"{self.log_dir}{self.run_name} already exists")

    def _setModel(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def _setDataset(self):
        if self.dataset_tag == "IndicHeadlineGeneration":
            self.traindataset, self.valdataset, self.testdataset = IndicHeadlineGenerationData(self.tokenizer)
        else:
            raise NotImplementedError("Dataset not implemented")
        
        self.datacollator = DataCollatorForSeq2Seq(
            self.tokenizer, 
            model=self.model,
            padding='max_length',
            max_length=256,
            )
        
    def _setTrainingArgs(self):
        self.training_args = Seq2SeqTrainingArguments(
            f"{self.run_name}",
            evaluation_strategy = "epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            save_total_limit=1,
            num_train_epochs=self.epochs,
            predict_with_generate=True,
            logging_dir=f"{self.log_dir}{self.run_name}",
            dataloader_num_workers=self.num_workers,
            load_best_model_at_end = True,
            save_strategy='epoch',
            logging_strategy='steps',
        )

    def _setMetric(self):
        self.metric = metricDic[self.metric]
    
    def _train(self):
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.datacollator,
            tokenizer=self.tokenizer,
            train_dataset=self.traindataset,
            eval_dataset=self.testdataset,
            compute_metrics=self.metric.compute
        )
        
        self.trainer.train()

    def run(self):
        self._setModel()
        self._setDataset()
        self._setTrainingArgs()
        self._setMetric()
        self._train()