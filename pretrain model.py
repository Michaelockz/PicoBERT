from model import PICOBERT, PICOBERTConfig

import torch
import torch.nn as nn
import torch.optim as optim
import os
from transformers import BertTokenizerFast, DataCollatorForLanguageModeling
from nltk.tokenize import sent_tokenize
import random
from torch.utils.data import IterableDataset, DataLoader


text_paths = os.listdir("D:/Datasets/persian wikipedia")
data_root = "D:/Datasets/persian wikipedia/"
file_paths = []
for path in text_paths:
    file_paths.append(os.path.join(data_root, path))


def multi_file_document_generator(file_paths):
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = line.strip()
                if doc:
                    sentence = sent_tokenize(doc)
                    if len(sentence) >= 1:
                        yield sentence


class MultiFileStreamingDataset(IterableDataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __iter__(self):
        return multi_file_document_generator(self.file_paths)

iter_dataset = MultiFileStreamingDataset(file_paths=file_paths)


class NSPMLMCollator:
    def __init__(self, tokenizer, mlm_probability):
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
        )
        self.tokenizer = tokenizer
        self.sep_token_id = tokenizer.sep_token_id

    def generate_nsp_pairs(self, examples):
        nsp_pairs = []
        nsp_labels = []
        max_attempts = 20
        for _ in range(len(examples)):
            is_next = random.random() > 0.5
            success = False
            attempts = 0

            while not success and attempts < max_attempts:
                attempts += 1
                try:
                    if is_next:
                        valid_docs = [d for d in examples if len(d) >= 2]
                        if not valid_docs:
                            raise ValueError("No valid docs for NSP=1")
                        doc = random.choice(valid_docs)
                        idx = random.randint(0, len(doc) - 2)
                        a, b = doc[idx], doc[idx + 1]
                        label = 1
                    else:
                        if len(examples) < 2:
                            raise ValueError("Not enough docs for NSP=0")
                        doc_a, doc_b = random.sample(examples, 2)
                        a = random.choice(doc_a) if doc_a else ""
                        b = random.choice(doc_b) if doc_b else ""
                        if not a or not b:
                            raise ValueError("Empty sentence")
                        label = 0

                    nsp_pairs.append((a, b))
                    nsp_labels.append(label)
                    success = True
                    del a, b

                except(IndexError, ValueError):
                    pass

            if not success:
                pass
        return nsp_pairs, nsp_labels

    def __call__(self, examples):

        nsp_pairs, nsp_labels = self.generate_nsp_pairs(examples)
        if not nsp_pairs:
            return None
        texts = [f"{a} {self.sep_token_id} {b}" for a, b in nsp_pairs]
        if not texts:
            return None
        tokenized = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        mlm_inputs = self.mlm_collator.torch_call(tokenized["input_ids"])

        return {
            "input_ids": mlm_inputs["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "token_type_ids": tokenized["token_type_ids"],
            "mlm_labels": mlm_inputs["labels"],
            "nsp_labels": torch.tensor(nsp_labels),
        }

tokenizer = BertTokenizerFast.from_pretrained('HooshvareLab/bert-fa-base-uncased')
mlm_probability = 0.15
collator = NSPMLMCollator(tokenizer, mlm_probability)
max_length = 512

train_loader = DataLoader(
    iter_dataset,
    batch_size=4,
    collate_fn=collator,
    num_workers=0,
    shuffle=False
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PICOBERT(PICOBERTConfig())
model.to(device)
optimizer = optim.AdamW(model.parameters(),lr=4e-5, fused=True)



device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("medium")
max_steps = 3500
total_batch_size = 128
B = 4
assert total_batch_size % B == 0
grad_accum_steps = total_batch_size // B
print(f"grad accum steps: {grad_accum_steps}")

from torch  .amp import GradScaler, autocast
scaler = GradScaler(device=device)

for step in range(max_steps):
    model.train()
    optimizer.zero_grad()
    loss_accum = 0
    for micro_step in range(grad_accum_steps):
        if micro_step % 6 == 0:
            print(f"\t\tproccesing micro batch: {micro_step} / {grad_accum_steps}")
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        mlm_labels = batch["mlm_labels"].to(device)
        nsp_labels = batch["nsp_labels"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mlm_logits, nsp_logits, loss = model(input_ids, attention_mask, token_type_ids, mlm_labels, nsp_labels)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    print(f"step: {step}, loss: {loss_accum:.6f}, norm: {norm:.4f}")
