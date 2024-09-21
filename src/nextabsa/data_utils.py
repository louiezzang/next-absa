import json
import yaml


def load_json_dataset(filepath):
    with open(filepath, "r") as file:
        data = json.load(file)
        return data
    

def load_instructions(filepath):
    with open(filepath, "r") as file:
        instructions = yaml.safe_load(file)
        return instructions
    

def create_data_in_aoste_format(data, instruction, term_key, polarity_key, text_col, aspect_col, opinion_col):
    label_map = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}

    bos_instruction = instruction["bos_instruct"]
    eos_instruction = instruction["eos_instruct"]

    for example in data:
        example["text"] = bos_instruction + example[text_col] + eos_instruction

        labels = []
        for aspect, opinion in zip(example[aspect_col], example[opinion_col]):
            label = f"{" ".join(aspect[term_key])}:{" ".join(opinion[term_key])}:{label_map[aspect[polarity_key]]}"
            labels.append(label)
        example["labels"] = ", ".join(labels)

    return data
