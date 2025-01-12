import sentencepiece as spm
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split

nltk.download('punkt')

def tokenize_english(text):
    """Tokenizes English text using nltk"""
    return nltk.word_tokenize(text.lower())

def tokenize_french(text, spm_model):
    """Tokenizes French text using SentencePiece model"""
    return spm_model.encode(text, out_type=str)

def prepare_data(file_path, spm_model_path):
    # Load SentencePiece model for French tokenization
    spm_model = spm.SentencePieceProcessor(model_file=spm_model_path)
    
    # Read the dataset from the Parquet file
    data = pd.read_parquet(file_path)
    
    # Ensure columns are named correctly
    if 'english' not in data.columns or 'french' not in data.columns:
        raise ValueError("Dataset must have 'english' and 'french' columns.")
    
    # Tokenize English and French text
    eng_sentences = data['english'].apply(tokenize_english).tolist()
    fr_sentences = data['french'].apply(lambda text: tokenize_french(text, spm_model)).tolist()
    
    # Split data into training (80%), validation (10%), and test sets (10%)
    eng_train, eng_temp, fr_train, fr_temp = train_test_split(eng_sentences, fr_sentences, test_size=0.2, random_state=42)
    eng_val, eng_test, fr_val, fr_test = train_test_split(eng_temp, fr_temp, test_size=0.5, random_state=42)

    return (eng_train, fr_train), (eng_val, fr_val), (eng_test, fr_test)