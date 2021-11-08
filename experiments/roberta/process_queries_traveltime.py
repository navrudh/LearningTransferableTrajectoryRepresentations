from transformers import AutoModel, PreTrainedTokenizerFast

from experiments.roberta.process_queries import process_queries

if __name__ == '__main__':
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="../../data/models/roberta/tokenizer-geohash-bbpe.json",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
    )

    for rate in [0.0, 0.2, 0.4, 0.6]:
        process_queries(
            query_file=f"../../data/train-transformer.test-ds-{rate}.query_database.pkl.csv",
            results_file=f"../../data/train-transformer.test-ds-{rate}.query_database.results.pkl",
            eval_model=AutoModel.from_pretrained('../../data/models/roberta/geohashcode-model'),
            tokenizer=tokenizer,
        )
