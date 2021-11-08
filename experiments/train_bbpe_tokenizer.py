from tokenizers.implementations import ByteLevelBPETokenizer

if __name__ == '__main__':
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=[
            "../data/train-transformer.train.dataframe.pkl.csv", "../data/train-transformer.val.dataframe.pkl.csv",
            "../data/train-transformer.test.query.pkl.csv", "../data/train-transformer.test.query_database.pkl.csv"
        ],
        vocab_size=52_000,
        min_frequency=2,
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ]
    )

    tokenizer.save("../data/tokenizer-geohash-bbpe.json")
