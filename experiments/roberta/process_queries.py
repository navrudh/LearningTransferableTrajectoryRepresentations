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
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="../../data/models/roberta/tokenizer-geohash-bbpe.json",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
    )

    process_queries(
        query_file="../../data/train-transformer.test.query.pkl.csv",
        results_file="../../data/train-transformer-h4.test.query.results.pkl",
        eval_model=AutoModel.from_pretrained('../../data/models/roberta/attention-head-4'),
        tokenizer=tokenizer,
    )
    process_queries(
        query_file="../../data/train-transformer.test.query_database.pkl.csv",
        results_file="../../data/train-transformer-h4.test.query_database.results.pkl",
        eval_model=AutoModel.from_pretrained('../../data/models/roberta/attention-head-4'),
        tokenizer=tokenizer,
    )
