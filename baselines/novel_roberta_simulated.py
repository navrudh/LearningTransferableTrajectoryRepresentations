import logging

from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling, PreTrainedTokenizerFast, RobertaConfig, RobertaForMaskedLM, Trainer,
    TrainingArguments
)

logger = logging.getLogger("experiment-logger")

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="../data/simulated/tokenizer-simulated-geohash-bbpe.json",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
    unk_token="<unk>",
    mask_token="<mask>",
)


def encode(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)


dataset = load_dataset(
    'text',
    data_files={
        'train': '../data/simulated/geohash/geohash.train.dataframe.pkl.csv',
        'validation': '../data/simulated/geohash/geohash_test.train.dataframe.pkl.csv'
    }
)
dataset.set_transform(encode)

config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,
    num_hidden_layers=3,
    max_position_embeddings=512,
    num_attention_heads=8,
    type_vocab_size=1
)

model = RobertaForMaskedLM(config=config)
logger.info("Model Parameters: %s", model.num_parameters())

logger.info("Dataset loaded")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)
logger.info("Data collator loaded")

training_args = TrainingArguments(
    output_dir="../data/simulated/models/roberta",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    remove_unused_columns=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)
logger.info("Begin training")
trainer.train()
logger.info("End training")
trainer.save_model("../data/simulated/models/roberta/attention-heads-8")
