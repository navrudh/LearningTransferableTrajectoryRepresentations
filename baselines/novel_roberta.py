import logging

from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling, PreTrainedTokenizerFast, RobertaConfig, RobertaForMaskedLM, Trainer,
    TrainingArguments
)

logger = logging.getLogger("experiment-logger")

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="../data/tokenizer-geohash-bbpe.json",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
    unk_token="<unk>",
    mask_token="<mask>",
)


def encode(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=200)


dataset = load_dataset(
    'text',
    data_files={
        'train': '../data/train-transformer.train.dataframe.pkl.csv',
        'validation': '../data/train-transformer.val.dataframe.pkl.csv'
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

model_path = "../data/models/roberta/v1-8epoch"
training_args = TrainingArguments(
    output_dir=model_path,
    overwrite_output_dir=True,
    num_train_epochs=8,
    per_device_train_batch_size=80,
    per_device_eval_batch_size=80,
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
trainer.save_model(f"{model_path}/final")
