from collections import Counter
import pandas as pd
import sys

def build_vocabulary(file_path):
    '''Build a vocabulary from the reviews in the given CSV file.'''

    df = pd.read_csv(file_path)

    all_text = ' '.join(df['review'].values) # Concatenate all reviews into a single string
    words = all_text.split() # Split the string into individual words

    word_counts = Counter(words) # Count the frequency of each word
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True) # type: ignore # Sort words by frequency (most common first)

    vocab_to_index = {word: idx + 1 for idx, word in enumerate(sorted_words)}
    return vocab_to_index

if __name__ == "__main__":
    file_path = sys.argv[1]
    vocab = build_vocabulary(file_path)
    print('Vocabulary size: {}'.format(len(vocab)))
    # Optionally, save the vocabulary to a file
    with open('./aclImdb/vocab.txt', 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write('{}\n'.format(word))
    print('Vocabulary saved to ./aclImdb/vocab.txt')