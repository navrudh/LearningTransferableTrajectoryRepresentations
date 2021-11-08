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

    process_queries(
        query_file="../../data/train-transformer.test-dest-traj.query.pkl.csv",
        results_file="../../data/train-transformer.test-dest-traj.query.results.pkl",
        eval_model=AutoModel.from_pretrained('../../data/models/roberta/geohashcode-model'),
        tokenizer=tokenizer,
    )
