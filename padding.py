import numpy as np

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
