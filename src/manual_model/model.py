from Transformer import Transformer
from CustomSchedule import CustomSchedule
from get_data import getData
from library import pd, np, tf, os, pickle, time, tqdm
from hyperParams_and_dirFile import (encoder_maxlen, decoder_maxlen, filters, oov_token, 
                                     BUFFER_SIZE, BATCH_SIZE, MODELS_DIR, EPOCHS, num_layers, 
                                     d_model, num_heads, dff, loss_object, train_loss, test_loss)
from datasets import load_metric


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

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
    return enc_padding_mask, combined_mask, dec_padding_mask

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


@tf.function
def train_step(inp, tar, model, optimizer):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = model(
            inp, tar_inp, 
            True, 
            enc_padding_mask, 
            combined_mask, 
            dec_padding_mask
        )
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)

@tf.function
def test_step(inp, tar, model):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    predictions, _ = model(
        inp, tar_inp, 
        False, 
        enc_padding_mask, 
        combined_mask, 
        dec_padding_mask
    )
    loss = loss_function(tar_real, predictions)
    test_loss(loss)

def training(dataset, total_train_batches, save_ckpt_manager):
    total_batches = len(dataset)

    for epoch in range(5):
        start = time.time()

        train_loss.reset_states()
        test_loss.reset_states()

        combined_bar = tqdm(enumerate(dataset), total=total_batches, desc=f'Epoch {epoch + 1} / {EPOCHS}')
        for batch, (inp, tar) in combined_bar:
            if batch < total_train_batches:
                # Training step
                train_step(inp, tar)
                combined_bar.set_postfix({'Train Loss': train_loss.result().numpy()}, refresh=True)
            else:
                # Test step
                test_step(inp, tar)
                combined_bar.set_postfix({'Train Loss': train_loss.result().numpy(), 'Test Loss': test_loss.result().numpy()}, refresh=True)

                
        print('Time taken for epoch {}: {} secs\n'.format(epoch + 1, time.time() - start))
        # Save checkpoint after each training epoch
        if (epoch + 1) % 2 == 0:
            ckpt_save_path = save_ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

def evaluate(input_document, model):
    path_document_tokenizer = os.path.join(MODELS_DIR, "document_tokenizer.pickle")
    path_summary_tokenizer = os.path.join(MODELS_DIR, "summary_tokenizer.pickle")
    with open(path_document_tokenizer, 'rb') as file:
        document_tokenizer = pickle.load(file)
    with open(path_summary_tokenizer, 'rb') as file:
        summary_tokenizer = pickle.load(file)
    
    input_document = document_tokenizer.texts_to_sequences([input_document])
    input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=encoder_maxlen, padding='post', truncating='post')

    encoder_input = tf.expand_dims(input_document[0], 0)

    decoder_input = [summary_tokenizer.word_index["[start]"]]
    output = tf.expand_dims(decoder_input, 0)
    for i in range(decoder_maxlen):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        predictions, attention_weights = model(
            encoder_input, 
            output,
            False,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask
        )

        predictions = predictions[: ,-1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == summary_tokenizer.word_index["[end]"]:
            return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

def summarize(input_document, model):
    path_summary_tokenizer = os.path.join(MODELS_DIR, "summary_tokenizer.pickle")
    with open(path_summary_tokenizer, 'rb') as file:
        summary_tokenizer = pickle.load(file)
    summarized = evaluate(input_document=input_document, model=model)[0].numpy()
    summarized = np.expand_dims(summarized[1:], 0)  # not printing <go> token
    return summary_tokenizer.sequences_to_texts(summarized)[0].replace('_', ' ')  # since there is just one translated document


def evaluate_rouge(datas, model):
    rouge_metric = load_metric('rouge')
    texts = datas['Contents']
    summaries = datas['Summary']
    references = []
    predictions = []
    for i in range(len(texts)):
        text = texts.iloc[i]
        actual = summaries.iloc[i]
        output = summarize(text, model)
        references.append(actual)
        predictions.append(output)
        print('.', end='', flush=True)
    rouge_metric.add_batch(predictions=predictions, references=references)
    rouge_scores = rouge_metric.compute()
    print(rouge_scores)
    return rouge_scores

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
    
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    # checkpoint
    save_checkpoint_path = os.path.join(MODELS_DIR, "checkpoints")
    save_ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

    save_ckpt_manager = tf.train.CheckpointManager(save_ckpt, save_checkpoint_path, max_to_keep=2)

    if save_ckpt_manager.latest_checkpoint:
        save_ckpt.restore(save_ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored from:', save_ckpt_manager.latest_checkpoint)

    ## training:
    # training(dataset=dataset, total_train_batches=len(train_dataset), save_ckpt_manager=save_checkpoint_path)
    
    ## summary:
    # print(summary(text, transformer))
    
    ## đánh giá độ chính xác bằng rouge
    # evaluate_rouge(datas=eval_data, model=transformer)

if __name__ == '__main__':
    main()