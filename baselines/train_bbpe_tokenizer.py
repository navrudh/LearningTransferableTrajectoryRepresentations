from tokenizers.implementations import ByteLevelBPETokenizer

if __name__ == '__main__':
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=[
            "../data/geohash_2/geohash.train.dataframe.pkl.csv",
            "../data/geohash_2/geohash.val.dataframe.pkl.csv",
            "../data/geohash_2/geohash.test.dataframe.pkl.csv",
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

    tokenizer.save("../data/models/tokenizer-geohash-bbpe.json")
