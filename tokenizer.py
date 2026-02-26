from vocabbuilder import build_vocabulary
import pandas as pd
import sys
import os

def tokenize_reviews(file_path, vocab_to_index):
    '''Tokenize the reviews in the given CSV file using the provided vocabulary.'''

    df = pd.read_csv(file_path)

    def tokenize(text):
        return [vocab_to_index.get(word, 0) for word in text.split()] # Use 0 for unknown words

    df['review'] = df['review'].apply(tokenize)
    return df[['review', 'sentiment', 'rating']]

if __name__ == "__main__":
    file_path = sys.argv[1]
    dest_path = os.path.join(os.path.dirname(os.path.dirname(file_path)), 'tokenized', os.path.basename(file_path).replace('.csv', '_tokenized.csv'))

    if os.path.exists('./aclImdb/vocab.txt'):
        with open('./aclImdb/vocab.txt', 'r', encoding='utf-8') as f:
            vocab = {line.strip(): idx + 1 for idx, line in enumerate(f)} # Load existing vocab
    else:
        vocab = build_vocabulary(file_path)

    tokenized_df = tokenize_reviews(file_path, vocab)
    tokenized_df.to_csv(dest_path, index=False)
    print(f'Tokenized dataset saved to {dest_path}')