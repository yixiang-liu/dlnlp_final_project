import warnings
warnings.filterwarnings('ignore')

from preprocess import multilang_en_nl_split

import torch
import numpy as np
import evaluate
from transformers import BertTokenizer,\
                        BertConfig,\
                        EncoderDecoderModel,\
                        EncoderDecoderConfig,\
                        DataCollatorForSeq2Seq,\
                        Seq2SeqTrainingArguments,\
                        Seq2SeqTrainer

# 加载模型
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',local_files_only=True)
print('tokenizer prepared!')

checkpoint = 'bert-base-multilingual-cased'
encoder_config = BertConfig.from_pretrained(checkpoint, device_map = torch.device('cuda:1'),local_files_only=True)
decoder_config = BertConfig.from_pretrained(checkpoint, device_map = torch.device('cuda:2'),local_files_only=True)
configs = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config,decoder_config)
model = EncoderDecoderModel.from_encoder_decoder_pretrained(checkpoint, checkpoint)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# 加载数据集
## 设置数据组织模式, 参数选择有:'random','en first','hu first'
mode = 'random'
tokenized_dataset = multilang_en_nl_split(checkpoint,how=mode,test_size=0.2)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

# 评估模型
bleu = evaluate.load('sacrebleu')
ter = evaluate.load('ter')
meteor = evaluate.load('meteor')

## 分词，便于bleu计算
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels
print('preparing metric!')

def compute_metrics(eval_preds):
    torch.cuda.empty_cache() 
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
    torch.cuda.empty_cache() 
    return result # dict
print('preparing model!')

# 设置训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./my_awesome_{mode}_{checkpoint}",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-05,
    weight_decay=0.05,

    num_train_epochs=0.05,
    logging_steps=200,
    save_steps=200,
    # eval_steps=200,
    evaluation_strategy='epoch',

    save_total_limit=3,
    fp16=True,
    dataloader_num_workers = 3,
    predict_with_generate=True,
)

# 初始化Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 训练模型
trainer.train()
