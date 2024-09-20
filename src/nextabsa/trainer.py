import numpy as np
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from transformers import (
    DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer
)


class AbsaTrainer:
    def __init__(self, model_checkpoint: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.device = "cuda" if torch.backends.cuda.is_built() else ("mps" if torch.backends.mps.is_built() else "cpu")

    @classmethod
    def tokenizer(cls):
        return cls.tokenizer
    
    def train(self, train_dataset, eval_dataset, **kwargs):
        """
        Trains the generative model.

        Args:
            datasets: datasets
        """
        # Set training arguments.
        args =  Seq2SeqTrainingArguments(**kwargs)

        # Define training object.
        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        print(f"Trainer device: {trainer.args.device}")

        # Finetune the model.
        torch.cuda.empty_cache()
        print("Model training started ...")
        trainer.train()

        # Save the best model.
        trainer.save_model()

        return trainer
