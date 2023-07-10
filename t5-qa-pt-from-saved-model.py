from datasets import load_dataset
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import transformers
import datasets
from transformers import AutoTokenizer, TFT5ForConditionalGeneration
import datetime
import os
import torch

model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained("t5-base")  

warmup_steps = 1e4
batch_size = 4
encoder_max_len = 250
decoder_max_len = 54
buffer_size = 1000
# ntrain = len(train_dataset)
# nvalid = len(valid_dataset)
# steps = int(np.ceil(ntrain/batch_size))
# valid_steps = int(np.ceil(nvalid/batch_size))
# print("Total Steps: ", steps)
# print("Total Validation Steps: ", valid_steps)

def encode(example,
           encoder_max_len=encoder_max_len, decoder_max_len=decoder_max_len):
  
    # from sample dataset 
    context = example['context']
    question = example['question']
    answer = example['answers']['text']
  
    # question and contexts are combined together 
    question_plus = f"answer_me: {str(question)}"
    question_plus += f" context: {str(context)} </s>"
    
    # 
    answer_plus = ', '.join([i for i in list(answer)])
    answer_plus = f"{answer_plus} </s>"
    
    encoder_inputs = tokenizer(question_plus, truncation=True, 
                               return_tensors='pt', max_length=encoder_max_len,
                              pad_to_max_length=True)
    
    decoder_inputs = tokenizer(answer_plus, truncation=True, 
                               return_tensors='pt', max_length=decoder_max_len,
                              pad_to_max_length=True)
    
    # input_ids and input_attention => input_ids and attention_mask
    input_ids = encoder_inputs['input_ids'][0]
    input_attention = encoder_inputs['attention_mask'][0]

    # target_ids => labels
    # target_attention => decoder_attention_mask
    target_ids = decoder_inputs['input_ids'][0]
    target_attention = decoder_inputs['attention_mask'][0]
    
    outputs = {'input_ids':input_ids, 'attention_mask': input_attention, 
               'labels':target_ids, 'decoder_attention_mask':target_attention}
    return outputs

train_size = 100

#accuracy metrics 
#metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy') ]

## Training 
epochs_done = 0
from transformers import T5ForConditionalGeneration


model_path = "./t5base-model-trained"

model = T5ForConditionalGeneration.from_pretrained(model_name)

model.load_state_dict(torch.load(model_path))
model.eval()

context = """Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary."""

question = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
print(context)
print(question)

input_text =  f"answer_me: {question} context: {context} </s>"
encoded_query = tokenizer(input_text, 
                         return_tensors='pt', pad_to_max_length=True, truncation=True, max_length=encoder_max_len)

input_ids = encoded_query["input_ids"]
attention_mask = encoded_query["attention_mask"]

generated_answer = model.generate(input_ids, attention_mask=attention_mask, 
                                 max_length=decoder_max_len, top_p=0.95, top_k=50, repetition_penalty=1)

#decoded_answer = tokenizer.decode(generated_answer.numpy()[0])
decoded_answer = tokenizer.decode(generated_answer[0])

decoded_answer2 = tokenizer.decode(generated_answer.numpy()[0])

print("Answer: ", decoded_answer)
print("Answer: ", decoded_answer2)

## training accuracy 
## eval the model - as in metrics - i need to be able to manipulate. it has an eval_loss by default - where is this coming from?


