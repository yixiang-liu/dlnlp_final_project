import warnings
warnings.filterwarnings('ignore')
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import numpy as np

# 加载数据
books = load_dataset("opus_books", "en-fr")
books = books["train"].train_test_split(test_size=0.2,seed=42)

# 可选的参数有:"t5-small","opus-mt-en-fr"
# 加载模型
checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)

# 确认语言
source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "

# 数据成对导入模型
def preprocess_t5_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

# if checkpoint == 't5-small':
tokenized_books = books.map(preprocess_t5_function, batched=True)
print('preparing datacollactor!')

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

# 评估模型
## 分词，便于bleu计算
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels
print('preparing metric!')

bleu = evaluate.load('sacrebleu')
ter = evaluate.load('ter')
meteor = evaluate.load('meteor')

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = {"bleu": bleu.compute(predictions=decoded_preds, references=decoded_labels)["score"],
              "ter": ter.compute(predictions=decoded_preds, references=decoded_labels)["score"],
              'meteor': meteor.compute(predictions=decoded_preds, references=decoded_labels)["meteor"],
              }

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result 
print('preparing model!')

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

print('preparing training!')
training_args = Seq2SeqTrainingArguments(
    output_dir=f"my_awesome_{checkpoint}",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    num_train_epochs=3,
    weight_decay=0.05,
    learning_rate=1e-3,
    evaluation_strategy ="epoch",

    logging_strategy = 'steps',
    logging_steps=100,
    save_steps=100,
    save_total_limit=3,

    predict_with_generate=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

print('training starts!')

trainer.train()
