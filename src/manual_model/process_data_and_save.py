from library import glob, os, pd, re, VnCoreNLP

# ! zip -r checkpoints.zip /data/dataset


rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.2.jar", annotators="wseg", max_heap_size='-Xmx2g')
translator = str.maketrans('', '', '!"“”#$&\'*+/:;<=>?@[\\]^_`{|}~\n')

def process_text(text: str):
    text = text.replace("\xa0", " ")
    text = re.sub(r'\S+@\S+', '', text)
    text = text.translate(translator)
    text = rdrsegmenter.tokenize(text)
    text = ' '.join([' '.join(x) for x in text])
    return text


def clean_data_to_save():
    folder_path = '../data/dataset_article'
    all_files = glob.glob(folder_path + "/*.csv")

    files_to_skip = ['Dataset_articles_NoID-5.csv','Dataset_articles_NoID-6.csv', 'Dataset_articles_NoID-7.csv'] # vì data nhiều nên bỏ qua một vài dataset
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
    document_proces.to_csv('../data/dataset_article/document_process.csv', index=False, encoding='utf-8')
    summary_proces.to_csv('../data/dataset_article/summary_process.csv', index=False, encoding='utf-8')
    dataframe = pd.concat([document_proces, summary_proces], axis=1)
    dataframe.to_csv('../data/dataset_article/data_process.csv', index=False)
    

def read_data_cleaned():
    data = pd.read_csv('../data/dataset_article/data_process.csv')

    data = data[data['Contents'].swifter.apply(lambda x: 50 < len(x.split()) <= 512)]
    data = data[data['Summary'].swifter.apply(lambda x: 5 < len(x.split()) <= 128)]
    data = data[data['Contents'].swifter.apply(lambda x: 50 < len(x.split()))]
    data = data[data['Summary'].swifter.apply(lambda x: 5 < len(x.split()))]
    return data

