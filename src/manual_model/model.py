from Transformer import Transformer
from get_data import getData
from library import pd, tf, os, pickle
from hyperParams_and_dirFile import (encoder_maxlen, decoder_maxlen, filters, oov_token, 
                                     BUFFER_SIZE, BATCH_SIZE, MODELS_DIR, num_layers, 
                                     d_model, num_heads, dff)





def createPipeDataset(data: pd.DataFrame):
    # texts to Sequence
    document = data['Contents']
    summary = data['Summary']

    summary = summary.apply(lambda x: '[START] ' + x + ' [END]')
    
    path_document_tokenizer = os.path.join(MODELS_DIR, "document_tokenizer.pickle")
    path_summary_tokenizer = os.path.join(MODELS_DIR, "summary_tokenizer.pickle")
    
    if os.path.exists(path_document_tokenizer):
        print(f"loading document_tokenizer...")
        with open(path_document_tokenizer, 'rb') as file:
            document_tokenizer = pickle.load(file)
        print(f"load {path_document_tokenizer} successful!!!\n")
    else:
        print(f"creating document_tokenizer...")
        document_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=filters, oov_token=oov_token)
        document_tokenizer.fit_on_texts(document)
        # saving
        with open(path_document_tokenizer, 'wb') as file:
            pickle.dump(document_tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Save {path_document_tokenizer} successful!!!\n")
        
    if os.path.exists(path_summary_tokenizer):
        print(f"loading summary_tokenizer...")
        with open(path_summary_tokenizer, 'rb') as file:
            summary_tokenizer = pickle.load(file)
        print(f"load {path_summary_tokenizer} successful!!!\n")
    else:
        print(f"creating summary_tokenizer...")
        summary_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=filters, oov_token=oov_token)
        summary_tokenizer.fit_on_texts(summary)
        # saving
        with open(path_summary_tokenizer, 'wb') as file:
            pickle.dump(summary_tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Save {path_summary_tokenizer} successful!!!\n")
   
    inputs = document_tokenizer.texts_to_sequences(document)
    targets = summary_tokenizer.texts_to_sequences(summary)

    # kích thước vocab encoder và decoder
    encoder_vocab_size = len(document_tokenizer.word_index) + 1
    decoder_vocab_size = len(summary_tokenizer.word_index) + 1
    
    # padding
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=encoder_maxlen, padding='post', truncating='post')
    targets = tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=decoder_maxlen, padding='post', truncating='post')

    inputs = tf.cast(inputs, dtype=tf.int32)
    targets = tf.cast(targets, dtype=tf.int32) 

    # tạo dataset
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    
    # chia train test (tỉ lệ 9:1)
    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    
    print("create Pipeline Dataset Successful!!!")
    
    return {'dataset': dataset,
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'encoder_vocab_size': encoder_vocab_size,
            'decoder_vocab_size': decoder_vocab_size}


def main():
    data = getData()
    pipelineDataset = createPipeDataset(data)
    dataset = pipelineDataset['dataset']
    train_dataset = pipelineDataset['train_dataset']
    test_dataset = pipelineDataset['test_dataset']
    encoder_vocab_size = pipelineDataset['encoder_vocab_size']
    decoder_vocab_size = pipelineDataset['decoder_vocab_size']
    
    transformer = Transformer(num_layers, 
                              d_model, 
                              num_heads, 
                              dff,
                              encoder_vocab_size, 
                              decoder_vocab_size, 
                              pe_input=encoder_vocab_size, 
                              pe_target=decoder_vocab_size)

if __name__ == '__main__':
    main()