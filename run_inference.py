from src.nextabsa.dataset import AbsaDataset
from src.nextabsa.generator import AbsaGenerator
import src.nextabsa.data_utils as data_utils


if __name__ == "__main__":
    model_checkpoint = "./output/checkpoints/laptops_opinion_aoste"
    generator = AbsaGenerator(model_checkpoint)
    tokenizer = generator.tokenizer
    
    # Prepare training dataset.
    instructions = data_utils.load_instructions("./instructions.yaml")
    test_data = data_utils.load_json_dataset("./datasets/SemEval14/Test/Laptops_Opinion_Test.json")
    test_data = test_data[:10]
    test_data = data_utils.create_data_in_aoste_format(
        test_data, 
        instruction=instructions["AOSTE"],
        term_key="term", 
        polarity_key="polarity", 
        text_col="raw_words",
        aspect_col="aspects", 
        opinion_col="opinions")
    
    test_dataset = AbsaDataset(
        data=test_data,
        tokenizer=tokenizer,
        source_col="text",
        source_max_length=512,
        train=False,
        verbose=True)
    
    preds = generator.generate(test_dataset, batch_size=4, max_length=128)
    for input, pred in zip(test_data, preds):
        print("\n", input["raw_words"], " ==> ", pred)
