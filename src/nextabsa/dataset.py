import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence, Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


IGNORE_INDEX = -100


def _tokenize_fn(strings: Sequence[str], tokenizer: Any, max_length: int) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: Any,
    max_length: int,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer, max_length) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class AbsaDataset(Dataset):

    def __init__(self,
                 data: Sequence[Dict],
                 tokenizer: Any,
                 max_length: int = 512,
                 verbose: bool = False
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        print("Tokenizing inputs... this may take some time...")
        # TODO: preprocess data

        self.input_ids = None
        self.labels = None


    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
