from library import os, VnCoreNLP, tf

# Các đường dẫn
CURR_DIR = os.getcwd() # đường dẫn hiện tại
PROJECT_DIR = os.path.abspath(os.path.join(os.path.join(CURR_DIR, os.pardir), os.pardir)) # đường dẫn của project
VNCORENLP_DIR = os.path.join(PROJECT_DIR, "vncorenlp\\VnCoreNLP-1.2.jar") # đường dẫn tới VNCoreNLP
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

DATA_DIR = os.path.join(PROJECT_DIR, 'data\\dataset_article') # đường dẫn tới folder dataset
FILE_DATA = os.path.join(DATA_DIR, 'data_process.csv') # đường dẫn tới file data dau khi xử lí
# print(VNCORENLP_DIR)

# xử lí token
rdrsegmenter = VnCoreNLP(VNCORENLP_DIR, annotators="wseg", max_heap_size='-Xmx2g')
translator = str.maketrans('', '', '!"“”#$&\'*+/:;<=>?@[\\]^_`{|}~\n')

filters = '!"#$%&()*+,-./:;<=>?@\\^`{|}~\t\n'
oov_token = '<unk>'

# đọ dài lớn nhất của encoder và decoder
encoder_maxlen = 512
decoder_maxlen = 128

# các siêu tham số
num_layers = 4
d_model = 256
dff = 512
num_heads = 8

EPOCHS = 20
BUFFER_SIZE = 20000
BATCH_SIZE = 64

# loss
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')