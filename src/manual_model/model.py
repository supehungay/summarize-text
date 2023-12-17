from Transformer import Transformer
from hyperParams import *
from library import os
from get_data import clean_data_to_save, load_data_cleaned, DATA_DIR

def getData():
    file_data = os.path.join(DATA_DIR, 'data_process.csv')
    # print(file_data)
    if os.path.exists(file_data):
        print(f"The file '{file_data}' exists. loading data_process.csv")
        data = load_data_cleaned()
        print(f"load {file_data} successful!!!")
        
    else:
        print(f"The file '{file_data}' does not exist. creating data_process.csv")
        clean_data_to_save()
        print(f"create {file_data} successful!!!")
        data = load_data_cleaned()
        print(f"load {file_data} successful!!!")
    return data

def get_tokenier():
    pass

def main():
#     transformer = Transformer(
#     num_layers, 
#     d_model, 
#     num_heads, 
#     dff,
#     encoder_vocab_size, 
#     decoder_vocab_size, 
#     pe_input=encoder_vocab_size, 
#     pe_target=decoder_vocab_size,
# )
    data = getData()
    print(data)

if __name__ == '__main__':
    main()