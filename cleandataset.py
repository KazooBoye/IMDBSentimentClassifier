import pandas as pd
import re
import sys

def clean_text(text):
    '''Clean the review text by converting to lowercase and removing punctuation.'''

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)   # Remove punctuation
    text = re.sub(r'<br\s*/?>', ' ', text)  # Replace HTML line breaks with space
    return text

def main():
    if len(sys.argv) != 2:
        print("Usage: python cleandataset.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    # Load the dataset
    df = pd.read_csv(file_path)

    # Clean the text data
    df['review'] = df['review'].apply(clean_text)

    # Save the cleaned dataset
    df.to_csv(file_path.replace('.csv', '_cleaned.csv'), index=False)
    print(f'Cleaned dataset saved to {file_path.replace(".csv", "_cleaned.csv")}')

if __name__ == "__main__":
    main()