import json
import yaml

import torch

from src.nextabsa.dataset import AbsaDataset
from src.nextabsa.generator import AbsaGenerator


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
    model_checkpoint = "./output/checkpoints/laptops_opinion_aoste"
    generator = AbsaGenerator(model_checkpoint)
    tokenizer = generator.tokenizer
    
    # Prepare training dataset.
    instructions = load_instructions("./instructions.yaml")
    test_data = load_json_dataset("./datasets/SemEval14/Test/Laptops_Opinion_Test.json")
    test_data = create_data_in_aoste_format(
        test_data, 
        term_key="term", 
        polarity_key="polarity", 
        aspect_col="aspects", 
        opinion_col="opinions")
    
    test_dataset = AbsaDataset(
        data=test_data,
        tokenizer=tokenizer,
        bos_instruction=instructions["AOSTE"]["bos_instruct"],
        eos_instruction=instructions["AOSTE"]["eos_instruct"],
        source_max_length=512,
        target_max_length=64,
        verbose=True)
    
    pred = generator.generate(test_dataset)
    print(pred)
