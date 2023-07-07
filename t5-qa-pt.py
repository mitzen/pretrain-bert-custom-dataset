from datasets import load_dataset
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import transformers
import datasets
from transformers import AutoTokenizer, TFT5ForConditionalGeneration
import datetime
import os

model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained("t5-base")  

train_dataset = load_dataset('squad', split='train')
valid_dataset = load_dataset('squad', split='validation')

print(train_dataset.features)

data = next(iter(train_dataset))
print("Example data from the dataset: \n", data)

warmup_steps = 1e4
batch_size = 4
encoder_max_len = 250
decoder_max_len = 54
buffer_size = 1000
ntrain = len(train_dataset)
nvalid = len(valid_dataset)
steps = int(np.ceil(ntrain/batch_size))
valid_steps = int(np.ceil(nvalid/batch_size))
print("Total Steps: ", steps)
print("Total Validation Steps: ", valid_steps)

def encode(example,
           encoder_max_len=encoder_max_len, decoder_max_len=decoder_max_len):
  
    context = example['context']
    question = example['question']
    answer = example['answers']['text']
  
    question_plus = f"answer_me: {str(question)}"
    question_plus += f" context: {str(context)} </s>"
    
    answer_plus = ', '.join([i for i in list(answer)])
    answer_plus = f"{answer_plus} </s>"
    
    encoder_inputs = tokenizer(question_plus, truncation=True, 
                               return_tensors='tf', max_length=encoder_max_len,
                              pad_to_max_length=True)
    
    decoder_inputs = tokenizer(answer_plus, truncation=True, 
                               return_tensors='tf', max_length=decoder_max_len,
                              pad_to_max_length=True)
    
    input_ids = encoder_inputs['input_ids'][0]
    input_attention = encoder_inputs['attention_mask'][0]
    target_ids = decoder_inputs['input_ids'][0]
    target_attention = decoder_inputs['attention_mask'][0]
    
    outputs = {'input_ids':input_ids, 'attention_mask': input_attention, 
               'labels':target_ids, 'decoder_attention_mask':target_attention}
    return outputs

train_ds =  train_dataset.map(encode)
valid_ds =  valid_dataset.map(encode)

## callback and metrics 

start_profile_batch = steps + 10
stop_profile_batch = start_profile_batch + 100
profile_range = f"{start_profile_batch},{stop_profile_batch}"

#accuracy metrics 
#metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy') ]

## Training 

epochs_done = 0
from transformers import T5ForConditionalGeneration, TrainingArguments, Trainer

model = T5ForConditionalGeneration.from_pretrained(model_name)

args = TrainingArguments(
    f"{model_name}-finetuned-t5",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
)

# tokenized_train = train_ds.map(encode, batched=True)
# tokenized_valid = valid_ds.map(encode, batched=True)


train_size = 100

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds.select(range(train_size)),
    eval_dataset=valid_ds.select(range(train_size)),
    #data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

context = """We went on a trip to Europe. We had our breakfast at 7 am in the morning at \
the nearby coffee shop. Wore a dark blue over coat for our first visit to Louvre Museum \
to experience history and art."""

question = "At what time did we had breakfast?"
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
