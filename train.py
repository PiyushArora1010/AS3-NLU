import os
import torch

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq

from module.data import IndicHeadlineGenerationData, IndicTranslationData
from module.models import getModel, getTokenizer
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

        try:
            os.makedirs(f"models/{self.run_name}")
        except:
            print(f"models/{self.run_name} already exists")


    def _setModel(self):
        self.model = getModel(self.model_name).to(device)
        self.tokenizer = getTokenizer(self.model_name)
        if 'bert' in self.model_name:
            self.model.decoder.decoder_start_token_id = self.tokenizer.cls_token_id
            self.model.decoder.bos_token_id = self.tokenizer.cls_token_id
            self.model.decoder.pad_token_id = self.tokenizer.pad_token_id
    
    def _setDataset(self):
        if self.dataset_tag == "IndicHeadlineGeneration":
            self.traindataset, self.valdataset, self.testdataset = IndicHeadlineGenerationData(self.tokenizer, self.samples)
        
        elif self.dataset_tag == "IndicTranslation":
            self.traindataset, self.valdataset, self.testdataset = IndicTranslationData(self.tokenizer, self.samples)

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
            f"models/{self.run_name}",
            evaluation_strategy = "epoch",
            optim="adamw_torch",
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
            overwrite_output_dir=True,
        )

    def _train(self):

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            decode_pred = self.tokenizer.batch_decode(predictions)
            decode_labels = self.tokenizer.batch_decode(labels)

            decoded_preds = ["\n".join(i.strip()) for i in decode_pred]
            decoded_labels = ["\n".join(i.strip()) for i in decode_labels]

            returnDic = {}
            for key, metric in metricDic.items():
                returnDic[key] = metric.compute(predictions=decoded_preds, references=decoded_labels)

            return returnDic
        
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.datacollator,
            tokenizer=self.tokenizer,
            train_dataset=self.traindataset,
            eval_dataset=self.valdataset,
            compute_metrics=compute_metrics,
        )
        
        self.trainer.train()

    def run(self):
        print("[Setting Model]")
        self._setModel()
        print("[Setting Dataset]")
        self._setDataset()
        print("[Setting Training Args]")
        self._setTrainingArgs()
        print("[Training]")
        self._train()
        print("[Training Done]")
