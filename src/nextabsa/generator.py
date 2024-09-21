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
