from library import glob, os, pd, re
from hyperParams_and_dirFile import translator, rdrsegmenter, DATA_DIR, FILE_DATA

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


def getData():
    file_data = os.path.join(DATA_DIR, 'data_process.csv')
    # print(file_data)
    if os.path.exists(file_data):
        print(f"The file '{file_data}' exists. loading data_process.csv\n")
        data = load_data_cleaned()
        print(f"\nload {file_data} successful!!!\n")
        
    else:
        print(f"The file '{file_data}' does not exist. creating data_process.csv")
        clean_data_to_save()
        print(f"\ncreate {file_data} successful!!!")
        data = load_data_cleaned()
        print(f"load {file_data} successful!!!\n")
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