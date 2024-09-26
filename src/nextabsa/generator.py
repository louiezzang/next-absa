from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM,
)


class AbsaGenerator:
    def __init__(self, model_checkpoint: str, tokenizer: AutoTokenizer = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint) if not tokenizer else tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.device = "cuda" if torch.backends.cuda.is_built() else ("mps" if torch.backends.mps.is_built() else "cpu")

    @classmethod
    def tokenizer(cls):
        return cls.tokenizer

    def generate(self, dataset, batch_size=4, max_length=128):
        def collate_fn(batch):
            input_ids = [torch.tensor(example["input_ids"]) for example in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return input_ids
        
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        predicted_output = []
        self.model.to(self.device)
        print("Model loaded to:", self.device)

        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            output_ids = self.model.generate(batch, max_length=max_length)
            output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for output_text in output_texts:
                predicted_output.append(output_text)
        return predicted_output

    def get_metrics(self, y_true, y_pred, is_triplet_extraction=False):
        total_pred = 0
        total_gt = 0
        tp = 0
        if not is_triplet_extraction:
            for gt, pred in zip(y_true, y_pred):
                gt_list = gt.split(", ")
                pred_list = pred.split(", ")
                total_pred+=len(pred_list)
                total_gt+=len(gt_list)
                for gt_val in gt_list:
                    for pred_val in pred_list:
                        if pred_val in gt_val or gt_val in pred_val:
                            tp+=1
                            break

        else:
            for gt, pred in zip(y_true, y_pred):
                gt_list = gt.split(", ")
                pred_list = pred.split(", ")
                total_pred+=len(pred_list)
                total_gt+=len(gt_list)
                for gt_val in gt_list:
                    gt_asp = gt_val.split(":")[0]

                    try:
                        gt_op = gt_val.split(":")[1]
                    except:
                        continue

                    try:
                        gt_sent = gt_val.split(":")[2]
                    except:
                        continue

                    for pred_val in pred_list:
                        pr_asp = pred_val.split(":")[0]

                        try:
                            pr_op = pred_val.split(":")[1]
                        except:
                            continue

                        try:
                            pr_sent = gt_val.split(":")[2]
                        except:
                            continue

                        if pr_asp in gt_asp and pr_op in gt_op and gt_sent == pr_sent:
                            tp+=1

        p = tp/total_pred
        r = tp/total_gt
        return p, r, 2*p*r/(p+r)
    