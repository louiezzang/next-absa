import json
import yaml

import torch

from src.nextabsa.dataset import AbsaDataset
from src.nextabsa.trainer import AbsaTrainer


def load_json_dataset(filepath):
    with open(filepath, "r") as file:
        data = json.load(file)
        return data
    

def load_instructions(filepath):
    with open(filepath, "r") as file:
        instructions = yaml.safe_load(file)
        return instructions
    

def create_data_in_aoste_format(data, term_key, polarity_key, aspect_col, opinion_col):
    label_map = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}

    for example in data:
        example["input"] = example["raw_words"]
        labels = []
        for aspect, opinion in zip(example[aspect_col], example[opinion_col]):
            label = f"{" ".join(aspect[term_key])}:{" ".join(opinion[term_key])}:{label_map[aspect[polarity_key]]}"
            labels.append(label)
        example["output"] = ", ".join(labels)

    return data


if __name__ == "__main__":
    pt_model_name = "allenai/tk-instruct-base-def-pos"
    trainer = AbsaTrainer(pt_model_name)
    tokenizer = trainer.tokenizer
    
    # Prepare training dataset.
    instructions = load_instructions("./instructions.yaml")
    train_data = load_json_dataset("./datasets/SemEval14/Train/Laptops_Opinion_Train.json")
    train_data = create_data_in_aoste_format(
        train_data, 
        term_key="term", 
        polarity_key="polarity", 
        aspect_col="aspects", 
        opinion_col="opinions")
    
    train_dataset = AbsaDataset(
        data=train_data,
        tokenizer=tokenizer,
        bos_instruction=instructions["AOSTE"]["bos_instruct"],
        eos_instruction=instructions["AOSTE"]["eos_instruct"],
        source_max_length=512,
        target_max_length=64,
        verbose=True)
    
    eval_data = load_json_dataset("./datasets/SemEval14/Validation/Laptops_Opinion_Validation.json")
    eval_data = create_data_in_aoste_format(
        eval_data, 
        term_key="term", 
        polarity_key="polarity", 
        aspect_col="aspects", 
        opinion_col="opinions")
    
    eval_dataset = AbsaDataset(
        data=eval_data,
        tokenizer=tokenizer,
        bos_instruction=instructions["AOSTE"]["bos_instruct"],
        eos_instruction=instructions["AOSTE"]["eos_instruct"],
        source_max_length=512,
        target_max_length=64,
        verbose=True)
    
    # Training arguments
    use_mps = True if torch.backends.mps.is_built() else False
    model_output_path = "./output/checkpoints/laptops_opinion_aoste"
    training_args = {
        "output_dir": model_output_path,
        "evaluation_strategy": "epoch",
        "learning_rate": 5e-5,
        "lr_scheduler_type": "cosine",
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 16,
        "num_train_epochs": 4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "save_strategy": "no",
        "load_best_model_at_end": False,
        "push_to_hub": False,
        "eval_accumulation_steps": 1,
        "predict_with_generate": True,
        "use_mps_device": use_mps
    }
    
    trainer.train(train_dataset, eval_dataset, **training_args)
