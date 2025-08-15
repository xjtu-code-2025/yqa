#和原来的相比：原来是固态padding，my_collate_fn自由度更高一些，可以是动态padding



from datasets import load_dataset
from transformers import AutoTokenizer
import torch

text_data = load_dataset("allenai/common_gen", split="train")
for i in range(4):
    print(text_data[i])

model_name = "openai/clip-vit-base-patch32"
tokenizer = AutoTokenizer.from_pretrained(model_name)



def add_eos_to_examples(example):
    string = ",".join(example['concepts'])  # "ski,mountain,skier"
    example['input_text'] = '%s .' % string
    example['target_text'] = '%s ' % example['target']
    return example

def my_collate_fn(batch):
    input_texts = [item['input_text'] for item in batch]
    target_texts = [item['target_text'] for item in batch]

    input_encodings = tokenizer(
        input_texts,
        padding=True,  
        truncation=True,
        return_tensors="pt"
    )

    target_encodings = tokenizer(
        target_texts,
        padding=True,  
        truncation=True,
        return_tensors="pt"
    ).input_ids

    labels_with_ignore_index = target_encodings.clone()
    labels_with_ignore_index[labels_with_ignore_index == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": labels_with_ignore_index
    }


text_data = text_data.map(add_eos_to_examples, batched=False)

text_loader = torch.utils.data.DataLoader(
    text_data,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    collate_fn=my_collate_fn 
)


try:
    for batch in text_loader:
        print({k: v.shape for k, v in batch.items()})  # 只打印 shape
        break
except Exception as e:
    print("error:", e)
