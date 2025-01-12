from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
from datasets import load_dataset

import warnings
warnings.filterwarnings('ignore')
# 参数设置
model_name = 'bert-base-multilingual-cased'
num_train_epochs = 3
train_batch_size = 16
eval_batch_size = 64
warmup_steps = 500
weight_decay = 0.01
logging_steps = 10

# 加载数据集
dataset = load_dataset("opus_books", "en-fr")
# print(dataset['train'][:1000])
dataset = dataset["train"].train_test_split(test_size=0.2)

# 初始化分词器
tokenizer = BertTokenizer.from_pretrained(model_name)

source_lang = 'en'
target_lang = 'fr'

# 数据预处理
def preprocess_function(examples):
    inputs = [example[source_lang] for example in examples["translation"]]
    # inputs = [example[source_lang] for example in examples["translation"]]

    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True,padding="max_length")
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 加载预训练的BERT模型
model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# 设置训练参数
training_args = TrainingArguments(
    output_dir="my_awesome_bert2bert",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    # logging_dir='./logs',
    logging_steps=logging_steps,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# 训练模型
trainer.train()
