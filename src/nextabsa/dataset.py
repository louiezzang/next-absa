from dataclasses import dataclass, field
from typing import Dict, Sequence, Any
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


def _tokenize_fn(strings: Sequence[str], tokenizer: Any, max_length: int) -> Dict:
    """Tokenize a list of strings."""

    tokenized_list = []
    for text in tqdm(strings, "tokenize"):
        tokenized = tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )
        tokenized_list.append(tokenized)

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
    source_max_length: int,
    target_max_length: int,
) -> Dict:
    """Preprocess the data by tokenizing."""
    
    sources_tokenized = _tokenize_fn(sources, tokenizer, source_max_length)
    targets_tokenized = _tokenize_fn(targets, tokenizer, target_max_length)
    return dict(input_ids=sources_tokenized["input_ids"], labels=targets_tokenized["input_ids"])


class AbsaDataset(Dataset):

    def __init__(self,
                 data: Sequence[Dict],
                 tokenizer: Any,
                 bos_instruction: str,
                 eos_instruction: str,
                 source_max_length: int = 512,
                 target_max_length: int = 64,
                 verbose: bool = False
                 ):
        super().__init__()

        self.tokenizer = tokenizer    
        sources = [bos_instruction.format_map(example) for example in data]
        targets = [eos_instruction.format_map(example) for example in data]

        if verbose:
            print((sources[0]))
            print((targets[0]))

        data_dict = preprocess(sources, targets, tokenizer, source_max_length, target_max_length)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]


    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
