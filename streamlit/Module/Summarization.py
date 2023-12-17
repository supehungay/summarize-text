from rouge import Rouge
import torch
import pytorch_lightning as pl
from transformers import AutoConfig
from transformers import (
    AdamW,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
import os
import numpy as np
from datasets import load_metric
import nltk
import re
from bs4 import BeautifulSoup

MODEL_NAME = "VietAI/vit5-base-vietnews-summarization"

class NewsSummaryModel(pl.LightningModule):
    def __init__(self, model_name, hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5):
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = hidden_dropout_prob
        self.config.attention_probs_dropout_prob = attention_probs_dropout_prob
    def __init__(self):
        super().__init__() # Khởi tạo lớp cha
        self.configuration = AutoConfig.from_pretrained(MODEL_NAME)
        self.configuration.hidden_dropout_prob = 0.5
        self.configuration.attention_probs_dropout_prob = 0.5
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, config = self.configuration)
                                                        
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels = None):
        output = self.model(
            input_ids, # Tensor chứa các ID token của văn bản.
            attention_mask=attention_mask, # Tensor mask cho biết token nào cần được chú ý trong quá trình encoding.
            labels=labels, # Tensor chứa các ID token của tóm tắt chính xác (chỉ được sử dụng trong quá trình huấn luyện).
            decoder_attention_mask=decoder_attention_mask
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids = input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss
    def validation_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids = input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss
    def test_step(self, batch, batch_idx):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids = input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0005)

class PretrainedSummary():
    def __init__(self) -> None:
        self.model = NewsSummaryModel()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    def load_from_checkpoint(self, path: str = ''):
        if (path==''):
            return
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def freeze(self):
        self.model.freeze()
    def summarize(self, text):
        # Mã hóa văn bản
        text_encoding = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        text_encoding = text_encoding.to(self.device)

        # Tạo tóm tắt
        generated_ids = self.model.model.generate(
            input_ids = text_encoding['input_ids'],
            attention_mask=text_encoding['attention_mask'],
            max_length=150,
            num_beams=2,
            repetition_penalty=1.0,
            early_stopping=True
        )
        # Giải mã tóm tắt
        preds = [
            self.tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for gen_id in generated_ids
        ]
        return "".join(preds) 
    
    @staticmethod
    def compute_rouge_scores(reference, summary):
        rouge = Rouge()
        scores = rouge.get_scores(summary, reference)
        return scores
    
    def evaluate(self, manual_summary, model_summary):
        scores = self.compute_rouge_scores(manual_summary, model_summary)
        for metric, value in scores[0]['rouge-l'].items():
            print(f'{metric}: {value}')
    
    def evaluate_rouge(self, datas):
        rouge_metric = load_metric('rouge')
        texts = datas['text']
        summaries = datas['summary']
        references = []
        predictions = []
        for i in range(len(texts)):
            text = texts.iloc[i]
            actual = summaries.iloc[i]
            output = self.summarize(text)
            references.append(actual)
            predictions.append(output)
        rouge_metric.add_batch(predictions=predictions, references=references)
        rouge_scores = rouge_metric.compute()
        print(rouge_scores)
        return rouge_scores

def process_text(text: str, pad: bool = True):
    # Remove email
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove https link
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove html tags for images, videos, etc.
    text = BeautifulSoup(text, "html.parser").get_text(separator=' ')
    
    # Replace multiple whitespaces with a single space and strip leading/trailing whitespaces
    text = re.sub('\s+', ' ', text).strip()

    if pad:
        return "vietnews: " + text + " </s>"
    return text