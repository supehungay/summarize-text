
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelPruning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from termcolor import colored
import textwrap
# from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap
from transformers import (
    AutoConfig,
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
from tqdm.auto import tqdm
import os
import re
from bs4 import BeautifulSoup

class NewsSummaryDataset(Dataset):
  def __init__(
      self,
      data: pd.DataFrame, # Pandas DataFrame chứa bài viết và bài tóm tắt
      tokenizer: AutoTokenizer, # Pre-trained tokenizer
      text_max_token_len: int = 512, # Chiều dài lớn nhất cho phép bài viết gốc có thể có
      summary_max_token_len: int = 128 # Chiều dài lớn nhất cho phép bài viết tóm tắt có thể có
  ):
    self.tokenizer = tokenizer
    self.data = data
    self.text_max_token_len = text_max_token_len
    self.summary_max_token_len = summary_max_token_len
  def __len__(self):
    return len(self.data) # Trả về số lượng phần tử trong dữ liệu
  def __getitem__(self, index: int): # Lấy dữ liệu theo chỉ mục
    data_row = self.data.iloc[index]
    text = data_row['text']
    text_encoding = self.tokenizer ( # Áp dụng tokenizer với cho văn bản gốc được lấy
        text, max_length=self.text_max_token_len,
        padding='max_length', # Padding thêm 0 vào mã hóa theo chiều dài lớn nhất
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    summary_encoding = self.tokenizer ( # Áp dụng tokenizer với cho văn bản tóm tắt được lấy
        data_row['summary'], max_length=self.summary_max_token_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    labels =summary_encoding['input_ids'] # Gán cacsn nhẵn trong danh sách đã được mã hóa thành với labels=0 thành -100
    labels[labels == 0] = -100
    # Ta có padding cho dữ liệu, nên sẽ tồn tại các token = 0 trong khi mã hóa
    # Nên ta cần gán các vị trí này là -100 cải thiện độ ổn định về mặt số học trong các phép tính
    # Ngoài ra hàm loss function sẽ chỉ xem xét các phần tử khác padding để tính toán giá trị loss

    return dict( # Trả về 1 từ điểm gồm
        text=text, # văn bản gốc
        summary=data_row['summary'], # văn bản tóm tắt
        text_input_ids =text_encoding['input_ids'].flatten(), # ID của văn bản gốc đã tokenizer
        text_attention_mask=text_encoding['attention_mask'].flatten(),  # ID của văn bản tóm tắt đã tokenizer
        labels=labels.flatten(),
        labels_attention_mask=summary_encoding['attention_mask'].flatten()
    )

class newsSummaryDataModule(pl.LightningDataModule):
  def __init__(
      self,
      train_df: pd.DataFrame, # DataFrame chứa dữ liệu huấn luyện (text và summary).
      test_df: pd.DataFrame,  # DataFrame chứa dữ liệu kiểm tra (text và summary).
      tokenizer: AutoTokenizer, # Tokenizer được sử dụng để xử lý text và tạo các biểu diễn số.
      batch_size: int = 8, # Kích thước batch được sử dụng trong quá trình huấn luyện và đánh giá.
      text_max_token_len: int = 512, # Chiều dài tối đa của input text.
      summary_max_token_len: int = 128 # Chiều dài tối đa của output summary.
  ):
    super().__init__()

    self.train_df = train_df
    self.test_df = test_df
    self.batch_size = batch_size
    self.tokenizer = tokenizer
    self.text_max_token_len = text_max_token_len
    self.summary_max_token_len = summary_max_token_len

  def setup(self, stage=None): # Thiết lập dữ liệu
    self.train_dataset = NewsSummaryDataset( # Khởi tạo một Instance của NewsSummaryDataset chứa dữ liệu đã được xử lý
        self.train_df,
        self.tokenizer,
        self.text_max_token_len,
        self.summary_max_token_len
    )

    self.test_dataset = NewsSummaryDataset( # Khởi tạo một Instance của NewsSummaryDataset chứa dữ liệu đã được xử lý
        self.test_df,
        self.tokenizer,
        self.text_max_token_len,
        self.summary_max_token_len
    )
  # Tạo DataLoader cho 3 tập dữ liệu train, valid, test
  def train_dataloader(self): # Trả về đối tượng DataLoader của dữ liệu
    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=2
    )
  def val_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=2
    )
  def test_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=2
    )

    # Đối tượng DataLoader được sử dụng để qu
    
class NewsSummaryModel(pl.LightningModule):
  def __init__(self):
    super().__init__() # Khởi tạo lớp cha
    # Tải mô hình T5 được đóng gói sẵn từ Hugging Face Model
    self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, return_dict=True)

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
    return torch.optim.Adam(self.parameters(), lr=0.001)