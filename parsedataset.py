import os
import pandas as pd

def load_to_dataframe(base_dir, split_name):
    """
    Load the IMDB dataset into a pandas DataFrame.
    base_dir: The base dataset directory
    split_name: The dataset split to load (train/test)
    """

    data = []
    for label in ['pos', 'neg']:
        # Path construct
        folder_path = os.path.join(base_dir, split_name, label)

        # Set binary sentiment label
        if label == 'pos': 
            sentiment = 1
        else:
            sentiment = 0
    
        #Iterate through all files
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):

                # Extract rating from filename
                rating_str = filename.split('_')[1].split('.')[0]
                rating = int(rating_str)

                # Read file contents
                with open (os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                    review_text = f.read()
                    data.append({'review': review_text, 'sentiment': sentiment, 'rating': rating})

    return pd.DataFrame(data)

if __name__ == "__main__":
    print('Loading and converting training set to CSV...')
    train_df = load_to_dataframe('./aclImdb', 'train')
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the training data
    train_df.to_csv('./aclImdb/parsed/train.csv', index=False)
    print('Done. Flushing dataframe...')
    del train_df  # Free memory after saving

    print('Loading and converting test set to CSV...')
    test_df = load_to_dataframe('./aclImdb', 'test')
    test_df.to_csv('./aclImdb/parsed/test.csv', index=False)
    print('Done. Flushing dataframe...')
    del test_df  # Free memory after saving

    print('All done')