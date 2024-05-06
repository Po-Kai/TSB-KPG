import random

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm


random.seed(42)


class NewsMKPDataset(Dataset):
    
    def __init__(self, path, *args, split="train", for_baseline=False, **kwargs):
        super().__init__(*args, **kwargs)
        
        langs = {
            "en": "en_XX",
            "fr": "fr_XX",
            "zh": "zh_CN",
            "ja": "ja_XX",
            "id": "id_ID",
            "my": "my_MM",
            "th": "th_TH",
            "vi": "vi_VN"
        }

        datasets = []
        lang_indices = {}
        offset = 0
        for lang, code in tqdm(langs.items()):
            tag = "low"
            if lang in ["en", "fr", "zh", "ja"]:
                tag = "high"

            data_path = f"{path}/{tag}-resources/{lang}/processed"

            dataset = load_dataset(
                "json", data_files=f"{data_path}/{split}.json",
            )["train"]
            
            dataset = self.preprocess(dataset, lang)
            start = offset
            end = offset + len(dataset)
            
            lang_indices[lang] = [idx for idx in range(start, end)]
            offset = end
            datasets.extend(dataset)
            
        self.datasets = datasets
        self.lang_indices = lang_indices
        self.split = split
        self.for_baseline = for_baseline
            
    def __getitem__(self, idx):
        lang, title, abstract, kps = self.datasets[idx]
        
        if not self.for_baseline and self.split == "train":
            while True:
                neg_sample_idx = random.choice(self.lang_indices[lang])
                neg_lang, neg_title, neg_abstract, _ = self.datasets[neg_sample_idx]
                assert lang == neg_lang
                if neg_title != title:
                    break
        
            return {
                "title": title,
                "abstract": abstract,
                "kps": kps,
                "negtive_abstract": neg_abstract
            }
        
        else:
            return {
                "title": title,
                "kps": kps,
            }
        

    def __len__(self):
        return len(self.datasets)

    def preprocess(self, examples, lang):
        pairs = []
        for title, abstract, present_kps, absent_kps in zip(
                examples["title"], examples["abstract"], examples["present_kps"], examples["absent_kps"]):

            title = title["text"]
            abstract = abstract["text"]
            kps = (";".join(present_kps["text"][:10] + absent_kps["text"][:10]))
            pairs.append((lang, title, abstract, kps))

        return pairs


class NewsMKPDataLoader(DataLoader):
    
    def __init__(self, *args, tokenizer, max_length=512, **kwargs):
        super(NewsMKPDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.tokenizer = tokenizer
        self.max_length = 256 # max_length
        
    def _collate_fn(self, batch):         
        titles = [sample["title"] for sample in batch]
        kpss = [sample["kps"] for sample in batch]
  
        model_inputs = self.tokenizer(titles, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        kpss = self.tokenizer(kpss, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        model_inputs["labels"] = kpss.input_ids
        
        if "abstract" in batch[0] and "negtive_abstract" in batch[0]:
            abstracts = [sample["abstract"] for sample in batch]
            negtive_abstracts = [sample["negtive_abstract"] for sample in batch]
        
            abstracts = self.tokenizer(abstracts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            negtive_abstracts = self.tokenizer(negtive_abstracts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            model_inputs["input_ids2"] = abstracts.input_ids
            model_inputs["attention_mask2"] = abstracts.attention_mask
            model_inputs["input_ids3"] = negtive_abstracts.input_ids
            model_inputs["attention_mask3"] = negtive_abstracts.attention_mask
        
        return model_inputs