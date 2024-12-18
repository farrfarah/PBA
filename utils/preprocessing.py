import pandas as pd

def preprocess_data(file_path, tokenizer, max_len=128):
    """
    Preprocess data by tokenizing 'title' and 'description' as input texts and returning labels.
    """
    data = pd.read_csv(file_path, header=None)  # No column headers
    texts = (data[1] + " " + data[2]).tolist()  # Combine 'title' and 'description'
    labels = (data[0] - 1).tolist()  # Subtract 1 from labels to make them 0-based

    # Tokenize using BERT tokenizer
    tokenized_texts = [
        tokenizer(text, truncation=True, padding='max_length', max_length=max_len)['input_ids']
        for text in texts
    ]
    return tokenized_texts, labels

