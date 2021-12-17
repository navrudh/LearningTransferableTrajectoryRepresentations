import os
import pickle

import torch
from tqdm import tqdm
from transformers import AutoModel, PreTrainedTokenizerFast


def read_txt_file(path: str):
    with open(path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    return lines


def process_queries(query_file, results_file, eval_model: AutoModel, tokenizer: PreTrainedTokenizerFast):
    """
    Save the output of the last hidden layer

    :param query_file: queries extracted from the test set
    :param results_file: file to save the hidden state
    :param model: the model that processes queries to results
    :param tokenizer: splits input into tokens
    :return: None
    """
    print("query_file:", query_file)

    test_results = []
    sentences = read_txt_file(query_file)

    for idx, sentence in enumerate(tqdm(sentences, desc="running query")):
        input_tokens = tokenizer.encode_plus(
            sentence, max_length=120, truncation=True, padding='max_length', return_tensors='pt'
        )

        outputs = eval_model(**input_tokens)
        embeddings = outputs.last_hidden_state
        # print('embeddings', embeddings.shape)

        attention_mask = input_tokens['attention_mask']
        # print('attention_mask', attention_mask.shape)

        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        # print('mask', mask.shape)

        masked_embeddings = embeddings * mask
        # print('masked_embeddings', masked_embeddings.shape)

        summed = torch.sum(masked_embeddings, 1)
        # print('summed', summed.shape)

        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        # print('summed_mask', summed_mask.shape)

        mean_pooled = summed / summed_mask
        # print('mean_pooled', mean_pooled.shape)

        test_results.append([idx + 1, mean_pooled.squeeze().detach().numpy()])

    pickle.dump(test_results, open(results_file, "wb"))


if __name__ == '__main__':
    data_dir = "../../data"
    experiment_data_dir = "geohash_1"

    model_path = '../../data/models/roberta/v1-8epoch/final'
    experiment_output_dir = "processed_roberta_h8"  # data_dir/outdir (created if missing)

    input_files = [
        # similarity
        "geohash.test.query.pkl.csv",
        "geohash.test.dataframe.pkl.csv",
        "geohash.test-similarity-ds_0.0.dataframe.pkl.csv",
        "geohash.test-similarity-ds_0.2.dataframe.pkl.csv",
        "geohash.test-similarity-ds_0.4.dataframe.pkl.csv",
        "geohash.test-similarity-ds_0.6.dataframe.pkl.csv",

        # destination prediction
        "geohash.test-dp-traj-ds_0.0.dataframe.pkl.csv",
        "geohash.test-dp-traj-ds_0.2.dataframe.pkl.csv",
        "geohash.test-dp-traj-ds_0.4.dataframe.pkl.csv",
        "geohash.test-dp-traj-ds_0.6.dataframe.pkl.csv",

        # travel time estimation
        "geohash.test-tte-ds_0.0.dataframe.pkl.csv",
        "geohash.test-tte-ds_0.2.dataframe.pkl.csv",
        "geohash.test-tte-ds_0.4.dataframe.pkl.csv",
        "geohash.test-tte-ds_0.6.dataframe.pkl.csv",
    ]

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="../../data/models/roberta/tokenizer-geohash-bbpe.json",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
    )

    _experiment_data_dir = f"{data_dir}/{experiment_data_dir}"
    _experiment_output_dir = f"{data_dir}/{experiment_output_dir}"
    # if Path(_experiment_output_dir).exists():
    #     print("Output directory already exists:", _experiment_output_dir)

    # eval_model = AutoModel.from_pretrained(model_path),

    os.makedirs(_experiment_output_dir, exist_ok=True)
    for src_file in input_files:
        dest_file = src_file.replace(".pkl.csv", ".results.pkl")
        process_queries(
            query_file=f"{_experiment_data_dir}/{src_file}",
            results_file=f"{_experiment_output_dir}/{dest_file}",
            eval_model=AutoModel.from_pretrained(model_path),
            tokenizer=tokenizer,
        )
