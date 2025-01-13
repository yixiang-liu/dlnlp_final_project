Welcome to the final project of Yixiang Liu and Zhenyu Guan. Our project focus on exploring how the construction of data impacts performance of LLM on
multi-language machine translation task. We devided the project into two parts: 
1. Compare the performance of different LLMs on Eng-to-Fra, including LSTM(RNN with attention), t5-samll, Opus-mt and bert2bert. Links to models mentioned
above are presented here. We select the best one as the object of the next part.
2. After part 1, we have got the best performing model is bert2bert. Now we decide to fine-tune bert2bert with multi-language translation data, which includes
Eng-Fra and Eng-Hun pairs. In this part, we compare randomly shuffled Eng-Fra and Eng-Hun pairs with Eng-Fra and Eng-Hun pairs appearing in turn.

+ Yixiang Liu is responsible for the idea genereation, writing training code, training t5-small, Opus-mt in part 1 and bert2bert in part 2 and part of report writing.
+ Zhenyu Guan is responsible for report writing, writing machine_translation_LSTM and machine_translation_bert and training RNN with attention and bert2bert in part 1.

Note: The parameter files are to large to be uploaded to Github, so if you are interested in our finetuned model, please download that at 
