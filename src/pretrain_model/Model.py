from PretrainedModelArchitect import *
import torch
N_EPOCHS = 20
BATCH_SIZE= 32
NUM_SAMPLE = 100000
from datasets import load_metric
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

    def transfer_learing(self):
        ''' We will freeze amount of model layer, only keep the last DECODE layer and FFN layer
        '''
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.model.decoder.block[11].layer[0].SelfAttention.q.weight.requires_grad= True
        self.model.model.decoder.block[11].layer[0].SelfAttention.k.weight.requires_grad= True
        self.model.model.decoder.block[11].layer[0].SelfAttention.v.weight.requires_grad= True
        self.model.model.decoder.block[11].layer[0].SelfAttention.o.weight.requires_grad= True
        self.model.model.decoder.block[11].layer[0].layer_norm.weight.requires_grad= True
        self.model.model.decoder.block[11].layer[1].EncDecAttention.q.weight.requires_grad= True
        self.model.model.decoder.block[11].layer[1].EncDecAttention.k.weight.requires_grad= True
        self.model.model.decoder.block[11].layer[1].EncDecAttention.v.weight.requires_grad= True
        self.model.model.decoder.block[11].layer[1].EncDecAttention.o.weight.requires_grad= True
        self.model.model.decoder.block[11].layer[1].layer_norm.weight.requires_grad= True
        self.model.model.decoder.block[11].layer[2].DenseReluDense.wi.weight.requires_grad= True
        self.model.model.decoder.block[11].layer[2].DenseReluDense.wo.weight.requires_grad= True
        self.model.model.decoder.block[11].layer[2].layer_norm.weight.requires_grad= True
        self.model.model.decoder.final_layer_norm.weight.requires_grad = True

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

def load_data(number_samples: int = NUM_SAMPLE, random_sate: int = 28, test_size: int = 0.01):
    dict_pandas = []
    for idx in range(1, 8):
        path = f'/content/Dataset_articles_NoID-{idx}.csv'
        temp = pd.read_csv(path)
        dict_pandas.append(temp)
    merged_dataframe = pd.concat(dict_pandas, ignore_index=True)
    merged_dataframe = merged_dataframe.dropna()
    df = merged_dataframe.sample(frac=1).reset_index(drop=True)[['Summary', 'Contents']]
    df.columns = ['summary', 'text']
    df = df.sample(n=NUM_SAMPLE, random_state=random_sate)
    df['text'] = df['text'].apply(process_text)
    train_df, test_df = train_test_split(df, test_size=test_size)
    return [train_df, test_df]

def callback(path:str = r"./model_saved", early_stopping_patient: int = 10):
    checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(path,"checkpoints"),
    filename='best_summary_checkpoint',
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min'
    )   

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=early_stopping_patient, verbose=True, mode="min")
    logger = TensorBoardLogger(os.path.join(path,"Tensorboard"), name='summary_vi')
    return [checkpoint_callback, early_stop_callback, logger]

def trainer():
    train_df, test_df = load_data(number_samples=NUM_SAMPLE, random_sate=28, test_size=0.01)
    data_module = newsSummaryDataModule(train_df, test_df, model.tokenizer, batch_size=BATCH_SIZE)
    checkpoint_callback, early_stop_callback, logger = callback()
    model = PretrainedSummary()
    model.transfer_learing()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)
    
    trainer = pl.Trainer(
    logger=logger,
    callbacks=[checkpoint_callback, early_stop_callback],
    max_epochs=N_EPOCHS,
    devices=1, accelerator= "gpu"
    )
    
    trainer.fit(model, data_module)
    
    path_save = '/kaggle/working/saved_model.pt'
    torch.save({
        'epoch': N_EPOCHS,
        'model_state_dict': trainer.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
    }, path_save)
    
    return model

    
def evaluate():
    _, test_df = load_data(NUM_SAMPLE=100000)
    trained_model = PretrainedSummary()
    trained_model.load_from_checkpoint('/kaggle/input/final-model/best_summary_checkpoint.ckpt')
    trained_model.freeze()
    rouge_scores = trained_model.evaluate_rouge(test_df)
    print("MID:\n")
    print("ROUGE1: " + str(rouge_scores['rouge1'].mid) + "\n")
    print("rouge2: " + str(rouge_scores['rouge2'].mid) + "\n")
    print("rougeL: " + str(rouge_scores['rougeL'].mid) + "\n")
    print("rougeLsum: " + str(rouge_scores['rougeLsum'].mid) + "\n")
    
    print("*" * 20)
    print("LOW:\n")
    print("ROUGE1: " + str(rouge_scores['rouge1'].low) + "\n")
    print("rouge2: " + str(rouge_scores['rouge2'].low) + "\n")
    print("rougeL: " + str(rouge_scores['rougeL'].low) + "\n")
    print("rougeLsum: " + str(rouge_scores['rougeLsum'].low) + "\n")
    
    print("*" * 20)
    print("HIGH:\n")
    print("ROUGE1: " + str(rouge_scores['rouge1'].high) + "\n")
    print("rouge2: " + str(rouge_scores['rouge2'].high) + "\n")
    print("rougeL: " + str(rouge_scores['rougeL'].high) + "\n")
    print("rougeLsum: " + str(rouge_scores['rougeLsum'].high) + "\n")
    
def main():
    ''' Here is an example code to start train model
    '''
    # trainer()
    
    ''' Here is an example code to evaluate model'''
    # evaluate()
    
if __name__ == "__main__":
    main()
    
    
    