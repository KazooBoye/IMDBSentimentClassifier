
# IMDB Sentiment Classifier

## Setup and Training Pipeline

### 1. Download and Parse Data
Download and parse data from train and test folders into CSV format:

```bash
./downloaddataset.sh  # Extract the dataset manually
python3 parsedataset.py
```

### 2. Clean Dataset
Standardize text, remove punctuation, and HTML line breaks:

```bash
python3 cleandataset.py "./aclImdb/parsed/train.csv"
python3 cleandataset.py "./aclImdb/parsed/test.csv"
```

### 3. Build Vocabulary
Create vocabulary from the training dataset:

```bash
python3 vocabbuilder.py "./aclImdb/parsed/train_cleaned.csv"
```

### 4. Tokenize Data
Tokenize using the built vocabulary:

```bash
python3 tokenizer.py "./aclImdb/parsed/train_cleaned.csv"
python3 tokenizer.py "./aclImdb/parsed/test_cleaned.csv"
```

### 5. Pad Data
Ensure all samples have uniform length:

```bash
python3 padding.py "./aclImdb/tokenized/train_cleaned_tokenized.csv"
python3 padding.py "./aclImdb/tokenized/test_cleaned_tokenized.csv"
```

### 6. Train Model
Run training (adjust `batch_size`, `embedding_dim`, `hidden_dim` as needed):

```bash
python3 batchtraining.py
```
