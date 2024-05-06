from datasets import load_dataset
from transformers import (
    BartTokenizerFast,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments, 
)

from models.bart_triplet_aa import BartForConditionalGeneration
from data_utils.datacollator import CustomDataCollatorForSeq2Seq


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
        titles, abstracts, kpss = [], [], []
        for title, abstract, present_kps, absent_kps in zip(
                examples["title"], examples["abstract"], examples["present_kps"], examples["absent_kps"]):

            title = title["text"]
            abstract = abstract["text"]
            kps = (";".join(present_kps["text"][:16] + absent_kps["text"][:16]))

            titles.append(title)
            abstracts.append(abstract)
            kpss.append(kps)


        model_inputs = tokenizer(
            titles, 
            max_length=max_length,
            truncation=True,
            padding=True
        )
        abstracts = tokenizer(
            abstracts, 
            max_length=max_length, 
            truncation=True,
            padding=True
        )
        with tokenizer.as_target_tokenizer():
            dec_inputs = tokenizer(
                kpss, 
                max_length=max_length, 
                truncation=True,
                padding=True
            )

        model_inputs["labels"] = dec_inputs.input_ids
        model_inputs["input_ids2"] = abstracts.input_ids
        model_inputs["attention_mask2"] = abstracts.attention_mask

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
    data_collator = CustomDataCollatorForSeq2Seq(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=f"save_models/bart/triplet-aa_scikp-5epoch",
        learning_rate=5e-5,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
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
