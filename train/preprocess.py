import warnings
warnings.filterwarnings('ignore')

from datasets import load_dataset,DatasetDict
from datasets.arrow_dataset import Dataset
from transformers import BertTokenizer
import pandas as pd

def multilang_en_nl_split(checkpoint, how, test_size = 0.2):
    # 加载数据
    fr_books = load_dataset("opus_books", "en-fr")
    hu_books = load_dataset("opus_books", "en-hu")
    fr_books = fr_books['train']
    hu_books = hu_books['train']
    tokenizer = BertTokenizer.from_pretrained(checkpoint, local_files_only=True)

    # 确认语言
    source_lang = "en"
    target1_lang = "fr"
    target2_lang = "hu"

    prefix1 = "translate English to French: "
    prefix2 = "translate English to Hungarian: "
    # 数据成对导入模型
    def preprocess_function(examples):
        try:
            inputs = [prefix1 + example[source_lang] for example in examples["translation"]]
            targets = [example[target1_lang] for example in examples["translation"]]
        except:
            inputs = [prefix2 + example[source_lang] for example in examples["translation"]]
            targets = [example[target2_lang] for example in examples["translation"]]
        
        model_inputs = tokenizer(inputs, text_target=targets, max_length=64, truncation=True, padding="max_length")
        return model_inputs

    # if checkpoint == 't5-small':

    tokenized_fr_books = fr_books.map(preprocess_function, batched=True)
    tokenized_hu_books = hu_books.map(preprocess_function, batched=True)

    fr_df = pd.DataFrame(tokenized_fr_books)
    hu_df = pd.DataFrame(tokenized_hu_books)

    if how == 'random':
        merged_df = pd.concat([fr_df,hu_df])
        merged_books = Dataset.from_pandas(merged_df)
        random_merged_books = merged_books.train_test_split(test_size,seed=42)

        return random_merged_books
    elif how == 'en first':
        l = int((1-test_size)*len(hu_df.index))
        unrandom_merged_train_df = pd.concat([fr_df.iloc[:l,:],hu_df.iloc[:l,:]])
        unrandom_merged_test_df = pd.concat([fr_df.iloc[l:len(hu_df.index),:],hu_df.iloc[l:len(hu_df.index),:]])
        unrandom_merged_books = DatasetDict({'train':Dataset.from_pandas(unrandom_merged_train_df),
                                            'test':Dataset.from_pandas(unrandom_merged_test_df)}) 
        
        return unrandom_merged_books
    elif how == 'hu first':
        l = int((1-test_size)*len(hu_df.index))
        unrandom_merged_train_df = pd.concat([hu_df.iloc[:l,:], fr_df.iloc[:l,:]])
        unrandom_merged_test_df = pd.concat([hu_df.iloc[l:len(hu_df.index),:],fr_df.iloc[l:len(hu_df.index),:]])
        unrandom_merged_books = DatasetDict({'train':Dataset.from_pandas(unrandom_merged_train_df),
                                            'test':Dataset.from_pandas(unrandom_merged_test_df)}) 

        return unrandom_merged_books
    else: 
        raise ValueError(f'how = {how} is not supported yet')


if __name__ == '__main__':
    data = multilang_en_nl_split(checkpoint='bert-base-multilingual-cased', how='random')
    print(data['train'][0])
