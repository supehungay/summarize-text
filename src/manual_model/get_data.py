from library import glob, os, pd, re, VnCoreNLP

# ! zip -r checkpoints.zip /data/dataset
CURR_DIR = os.getcwd() # đường dẫn hiện tại
PROJECT_DIR = os.path.abspath(os.path.join(os.path.join(CURR_DIR, os.pardir), os.pardir)) # đường dẫn của project
DATA_DIR = os.path.join(PROJECT_DIR, 'data\\dataset_article') # đường dẫn tới folder dataset
FILE_DATA = os.path.join(DATA_DIR, 'data_process.csv') # đường dẫn tới file data dau khi xử lí
VNCORENLP_DIR = os.path.join(PROJECT_DIR, "vncorenlp\\VnCoreNLP-1.2.jar") # đường dẫn tới VNCoreNLP
# print(VNCORENLP_DIR)

rdrsegmenter = VnCoreNLP(VNCORENLP_DIR, annotators="wseg", max_heap_size='-Xmx2g')
translator = str.maketrans('', '', '!"“”#$&\'*+/:;<=>?@[\\]^_`{|}~\n')

def process_text(text: str):
    text = text.replace("\xa0", " ")
    text = re.sub(r'\S+@\S+', '', text)
    text = text.translate(translator)
    text = rdrsegmenter.tokenize(text)
    text = ' '.join([' '.join(x) for x in text])
    return text


def clean_data_to_save():
    all_files = glob.glob(DATA_DIR + "\\*.csv")

    files_to_skip = ['Dataset_articles_NoID-1.csv', 'Dataset_articles_NoID-2.csv', 'Dataset_articles_NoID-3.csv', 'Dataset_articles_NoID-4.csv', 'Dataset_articles_NoID-5.csv','Dataset_articles_NoID-6.csv'] # vì data nhiều nên bỏ qua một vài dataset
    list_data = []
    for file in all_files:
        if os.path.basename(file) in files_to_skip:
            continue
        df = pd.read_csv(file)
        list_data.append(df)

    data = pd.concat(list_data, ignore_index=True)
    data.drop(['URL', 'Title', 'Date', 'Author(s)', 'Category', 'Tags'], axis=1, inplace=True)
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    
    document_proces = data['Contents'].swifter.apply(process_text)
    summary_proces = data['Summary'].swifter.apply(process_text)
    dataframe = pd.concat([document_proces, summary_proces], axis=1)
    dataframe.to_csv(FILE_DATA, index=False)
    

def load_data_cleaned():
    data = pd.read_csv(FILE_DATA)

    data = data[data['Contents'].swifter.apply(lambda x: 50 < len(x.split()) <= 512)]
    data = data[data['Summary'].swifter.apply(lambda x: 5 < len(x.split()) <= 128)]
    data = data[data['Contents'].swifter.apply(lambda x: 50 < len(x.split()))]
    data = data[data['Summary'].swifter.apply(lambda x: 5 < len(x.split()))]
    return data

# def main():
    
#     file_data = os.path.join(DATA_DIR, 'data_process.csv')
#     # print(file_data)
#     if os.path.exists(file_data):
#         print(f"The file '{file_data}' exists. read data_process.csv")
#         # data = load_data_cleaned()
#     else:
#         print(f"The file '{file_data}' does not exist. creating data_process.csv")
#         # clean_data_to_save()

# if __name__ == '__main__':
#     main()