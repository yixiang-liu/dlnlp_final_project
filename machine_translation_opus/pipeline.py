import warnings 
warnings.filterwarnings('ignore')
from transformers import pipeline, AutoTokenizer


# 在这里输入英语
source_language = 'Family, who gets it? Trained an AI model, and the accuracy ended up worse than random guessing!'
# 在这里输入目标语言种类，目前支持France, Hungarian
checkpoint = 'machine_translation_t5_small\\opus'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print('translator preparing!')
translator = pipeline('translation',model=checkpoint,tokenizer=tokenizer,max_length=32)
print('translator prepared!')
english_text = f"Translate English into France:{source_language}"

# 进行翻译
print('translation starts!')
translated_text = translator(english_text)

# 打印翻译结果
print("Translated Text:", translated_text[0]['translation_text'])