import numpy as np
import torch
import torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer
)


class AbsaGenerator:
    def __init__(self, model_checkpoint: str, tokenizer: AutoTokenizer = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint) if not tokenizer else tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.device = "cuda" if torch.has_cuda else ("mps" if torch.has_mps else "cpu")

    def generate(self, dataset, batch_size=4, max_length=128):
        pass
