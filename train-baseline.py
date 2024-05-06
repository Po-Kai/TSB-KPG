from datasets import load_dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizerFast,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments, 
)



tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
special_tokens_dict = {"additional_special_tokens": ["[digit]"]}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)


def model_init():
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    model.resize_token_embeddings(len(tokenizer))
    return model


def get_dataset():
    data_root = "data/scikp/kp20k/processed/"
    dataset = load_dataset(
        "json", data_files={"train": f"{data_root}/train.json", "validation": f"{data_root}/valid.json"}
    )


    len_filter = lambda example: len(example["title"]["text"]) > 0 and len(example["abstract"]["text"]) > 0 and (len(example["present_kps"]["text"]) > 0 or len(example["absent_kps"]["text"]) > 0)


    dataset = dataset.filter(len_filter, num_proc=16)

    def preprocess_function(examples, max_length=512):
        titles, kpss = [], []
        for title, abstract, present_kps, absent_kps in zip(
                examples["title"], examples["abstract"], examples["present_kps"], examples["absent_kps"]):

            title = title["text"]
            kps = (";".join(present_kps["text"][:16] + absent_kps["text"][:16]))

            titles.append(title)
            kpss.append(kps)


        model_inputs = tokenizer(
            titles, 
            max_length=max_length,
            truncation=True,
            padding=True
        )

        dec_inputs = tokenizer(
            kpss,
            max_length=max_length, 
            truncation=True,
            padding=True
        )

        model_inputs["labels"] = dec_inputs.input_ids

        return model_inputs


    dataset = dataset.map(
        preprocess_function, 
        remove_columns=["id", "title", "abstract", "present_kps", "absent_kps"], 
        batched=True, 
        num_proc=16
    )

    return dataset

    
if __name__ == "__main__":
    model = model_init()
    dataset = get_dataset()
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=f"save_models/bart/baseline_scikp",
        learning_rate=5e-5,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        save_strategy="epoch",
        dataloader_num_workers=4,
        gradient_checkpointing=False,
        seed=2023,
        data_seed=2023,
        report_to="tensorboard",
        ddp_find_unused_parameters=False
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    trainer.train()
