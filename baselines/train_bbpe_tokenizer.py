from tokenizers.implementations import ByteLevelBPETokenizer

if __name__ == '__main__':
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=[
            "../data/simulated/train.train.dataframe.pkl.csv",
            "../data/simulated/test.train.dataframe.pkl.csv",
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

    tokenizer.save("../data/simulated/tokenizer-geohash-bbpe.json")
