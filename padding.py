import numpy as np
import pandas as pd
import sys
import ast

def pad_features(reviews, seq_length):
    '''Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.'''

    features = np.zeros((len(reviews), seq_length), dtype=int) # Create empty array of review_ints, of shape (len(reviews), seq_length)

    for i, review in enumerate(reviews): # Loop through review_ints, should be a list of lists
        review_len = len(review)
        if review_len <= seq_length:
            features[i, -review_len:] = review # If review is too short, pad with 0's at the front
        else:
            features[i, :] = review[:seq_length] # If review is too long, truncate and keep only the first seq_length words

    return features

seq_length = 200 # Standard length for reviews (in words)

if __name__ == "__main__":
    print('Padding features...')

    file_path = sys.argv[1]
    df = pd.read_csv(file_path)
    reviews = df['review'].apply(ast.literal_eval) # Convert string representation of list back to list
    features = pad_features(reviews, seq_length)
    np.save(file_path.replace('.csv', '_features.npy'), features)
    print(f'Features saved to {file_path.replace(".csv", "_paddedfeatures.npy")}')
