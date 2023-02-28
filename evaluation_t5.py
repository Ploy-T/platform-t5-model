from __future__ import print_function
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import DataLoader 
import argparse
from MyDataset import DataSet
from datasets import Dataset
import MyDataset

dataset_instruction = {
    "duorc": {
        "parser": MyDataset.DatasetMap.duorc,
        "test_set": "test"
    },
    "squad": {
        "parser": MyDataset.DatasetMap.squad,
        "test_set": "validation"
    }
}

def parse_command_line_arguments():

    parser = argparse.ArgumentParser(
        description='CLI for evaluating T5 T2T model')

    parser.add_argument('--t5_model', type=str, default="results/t5-base/checkpoint-31",
                        help="What type of T5 model do you want use?")

    parser.add_argument('--dataset', type=str, default='duorc-SelfRC',
                        help="Dataset to be used, if more level provided for the dataset use the '-' token, e.g. duorc-SelfRC")
    
    parser.add_argument('--batch_size', type=int, default=16,
                        help='mini-batch size (default: 16)')
    
    parser.add_argument('--workers', type=int, default=10,
                        help='number of working units used to load the data (default: 10)')

    parser.add_argument('--device', default='cuda', type=str,
                        help='device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cpu)')

    parser.add_argument('--max_input_length', type=int, default=512,
                        help='Maximum lenght of input text, (default: 512, maximum admitted: 512)')

    parser.add_argument('--seed', type=int, default=7,
                        help='Seed for random initialization (default: 7)')

    parsed_arguments = parser.parse_args()

    return parsed_arguments


if __name__ == '__main__':
    #args = parse_command_line_arguments()

    # for k, v in args.__dict__.items():
    #     print(k + '=' + str(v))

    # dataset_info = args.dataset.split("-")
    # name = dataset_info[0]
    # _data = None
    # if len(dataset_info) == 1:
    #     _data = load_dataset(name)
    # else:
    #     _data = load_dataset(name, dataset_info[1])

    model = 't5-base'
    tokenizer = 't5-base'
    batch_size = 1
    device = 'cpu'
    max_input_length = 512
        
    model = T5ForConditionalGeneration.from_pretrained(model)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer)
    
    _data = Dataset.from_json('EffectiveDateDatasetSplitted.json')
    _data
    print('---------------')
    print(_data)
    _data = _data.train_test_split(test_size=0.1)
    print('-----------------')
    #_test_set = Dataset(_data[dataset_instruction[name]["test_set"]], tokenizer, parser=dataset_instruction[name]["parser"])
    
    _test_set = DataSet(_data['test'],tokenizer, parser=dataset_instruction['squad']["parser"])
    #_test_set = DataSet(_data, tokenizer, parser=dataset_instruction['squad']["parser"])

    my_testset_dataloader = DataLoader(_test_set, batch_size=batch_size, num_workers=1, collate_fn=lambda data: _test_set.pack_minibatch(data))
    
    device = device
    model.to(device)

    model.eval()
    with torch.no_grad():
        model_predictions_encoded = []
        target_encoded = []
        for contexts, questions, answers in tqdm(my_testset_dataloader):
            inputs = list(map(lambda tuple: f"question: {tuple[0]}  context:{tuple[1]}", zip(
                questions, contexts)))
            encoded_inputs = tokenizer(
                inputs,
                padding="longest",
                max_length=max_input_length,
                truncation=True,
                return_tensors="pt",
            )
            encoded_targets = tokenizer(
                answers,
                padding="longest",
                max_length=max_input_length,
                truncation=True,
                return_tensors="pt",
            )
            encoded_inputs, attention_mask = encoded_inputs.input_ids, encoded_inputs.attention_mask
            encoded_targets = encoded_targets.input_ids

            encoded_inputs = encoded_inputs.to(device)
            encoded_targets = encoded_targets.to(device)
            attention_mask = attention_mask.to(device)
            model_predictions = model.generate(
                input_ids=encoded_inputs, attention_mask=attention_mask)

            model_predictions_encoded += model_predictions.tolist()
            target_encoded += encoded_targets.tolist()
            
    f1, exact_match = _test_set.evaluate(
        model_predictions_encoded, target_encoded)
    print(f"\t F1 = {f1:.2f}, EM = {exact_match:.2f}")
