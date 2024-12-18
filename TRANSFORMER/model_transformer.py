import pandas as pd
import numpy as np
import re
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from nltk.corpus import stopwords

# Pastikan nltk sudah di-download
nltk.download('stopwords')

# Fungsi untuk preprocessing teks
def preprocess_text(text, stop_words):
    text = re.sub(r'http\S+', '', text)  # Hapus URL
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hapus karakter selain huruf
    text = text.lower()  # Ubah ke lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Hapus stopwords
    return text

# Fungsi untuk membangun model transformer
def build_transformer_model(input_dim, embedding_dim, embedding_matrix, max_len):
    # Input layer
    inputs = layers.Input(shape=(max_len,))
    
    # Embedding Layer dengan Pre-trained GloVe Embedding
    embedding_layer = Embedding(input_dim=input_dim, 
                                output_dim=embedding_dim, 
                                weights=[embedding_matrix], 
                                input_length=max_len, 
                                trainable=False)(inputs)
    
    # Transformer block
    x = layers.MultiHeadAttention(num_heads=2, key_dim=embedding_dim)(embedding_layer, embedding_layer)
    x = LayerNormalization()(x)
    x = Dropout(0.1)(x)
    
    # Pooling
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers untuk klasifikasi
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(4, activation='softmax')(x)  # 4 output classes (sesuaikan dengan jumlah kelas di data)

    # Model
    model = models.Model(inputs, outputs)
    return model

# Fungsi untuk mempersiapkan data dan melatih model
def train_model(train_file, test_file, glove_file, max_words=10000, max_len=100, embedding_dim=100, batch_size=512, epochs=3):
    # Membaca Data
    train_data = pd.read_csv(train_file, header=None)
    test_data = pd.read_csv(test_file, header=None)

    # X adalah kolom 'combined' dalam bentuk list
    X_train = train_data[1].tolist()
    X_test = test_data[1].tolist()

    # y adalah kolom ke-0 dalam bentuk list
    y_train = train_data[0].tolist()
    y_test = test_data[0].tolist()

    stop_words = set(stopwords.words('english'))

    # Preprocessing Data (Membersihkan Teks)
    X_train = [preprocess_text(text, stop_words) for text in X_train]
    X_test = [preprocess_text(text, stop_words) for text in X_test]

    # Tokenisasi
    tokenizer = Tokenizer(num_words=max_words, lower=True)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

    # Label Encoding
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    # Memuat Pre-trained GloVe Embedding
    embedding_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vector

    # Membuat Matriks Embedding
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i < max_words:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    # Menyiapkan model
    model = build_transformer_model(input_dim=max_words, 
                                    embedding_dim=embedding_dim, 
                                    embedding_matrix=embedding_matrix, 
                                    max_len=max_len)

    # Menyusun model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # Melatih model
    history = model.fit(X_train_pad, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(X_test_pad, y_test))

    return model, history
