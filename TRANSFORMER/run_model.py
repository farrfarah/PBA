from model_transformer import train_model

# Tentukan file data dan GloVe
train_file = '/content/drive/MyDrive/ag_news_csv/train.csv'
test_file = '/content/drive/MyDrive/ag_news_csv/test.csv'
glove_file = '/content/drive/MyDrive/glove.6B.100d.txt'

# Menjalankan fungsi untuk melatih model
model, history = train_model(train_file, test_file, glove_file)

# Menampilkan hasil pelatihan
print("Model berhasil dilatih!")
