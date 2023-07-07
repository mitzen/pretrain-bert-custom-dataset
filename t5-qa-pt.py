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

data_dir = "./data"
log_dir = f"{data_dir}/experiments/t5/logs"
save_path = f"{data_dir}/experiments/t5/models"
cache_path_train = f"{data_dir}/cache/t5.train"
cache_path_test = f"{data_dir}/cache/t5.test"

tokenizer = AutoTokenizer.from_pretrained("t5-base")  
train_dataset = load_dataset('squad', split='train')
valid_dataset = load_dataset('squad', split='validation')

train_dataset.features

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

#
# input_ids, attention_mask, labels, decoder_attention_mask

# ex = next(iter(train_ds))
# print("Example data from the mapped dataset: \n", ex)

# def to_tf_dataset(dataset):  
#   columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']
#   dataset.set_format(type='tensorflow', columns=columns)
#   return_types = {'input_ids':tf.int32, 'attention_mask':tf.int32, 
#                 'labels':tf.int32, 'decoder_attention_mask':tf.int32,  }
#   return_shapes = {'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]), 
#                   'labels': tf.TensorShape([None]), 'decoder_attention_mask':tf.TensorShape([None])}
#   ds = tf.data.Dataset.from_generator(lambda : dataset, return_types, return_shapes)
#   return ds

# tf_train_ds = to_tf_dataset(train_ds)
# tf_valid_ds = to_tf_dataset(valid_ds)


# def create_dataset(dataset, cache_path=None, batch_size=4, 
#                    buffer_size= 1000, shuffling=True):    
#     if cache_path is not None:
#         dataset = dataset.cache(cache_path)        
#     if shuffling:
#         dataset = dataset.shuffle(buffer_size)
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
#     return dataset

# tf_train_ds= create_dataset(tf_train_ds, batch_size=batch_size, 
#                          shuffling=True, cache_path = None)
# tf_valid_ds = create_dataset(tf_valid_ds, batch_size=batch_size, 
#                          shuffling=False, cache_path = None)

# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#   def __init__(self, warmup_steps=1e4):
#     super().__init__()

#     self.warmup_steps = tf.cast(warmup_steps, tf.float32)
    
#   def __call__(self, step):
#     step = tf.cast(step, tf.float32)
#     m = tf.maximum(self.warmup_steps, step)
#     m = tf.cast(m, tf.float32)
#     lr = tf.math.rsqrt(m)
#     return lr 
  

## callback and metrics 

start_profile_batch = steps + 10
stop_profile_batch = start_profile_batch + 100
profile_range = f"{start_profile_batch},{stop_profile_batch}"

log_path = log_dir + "/" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1,
#                                                     update_freq=20,profile_batch=profile_range)

# checkpoint_filepath = save_path + "/" + "T5-{epoch:04d}-{val_loss:.4f}.ckpt"
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=False,
#     monitor='val_loss',
#     mode='min',
#     save_best_only=True)

#callbacks = [tensorboard_callback, model_checkpoint_callback] 
metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy') ]

## Training 
#learning_rate = CustomSchedule()
# learning_rate = 0.001  # Instead set a static learning rate
#optimizer = tf.keras.optimizers.Adam(learning_rate)

# model = SnapthatT5.from_pretrained("t5-base")
# model.compile(optimizer=optimizer, metrics=metrics)

epochs_done = 0
# model.fit(tf_train_ds, epochs=5, steps_per_epoch=steps, callbacks=callbacks, 
#           validation_data=tf_valid_ds, validation_steps=valid_steps, initial_epoch=epochs_done)

#model.save_pretrained(save_path)
### Testing model

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
    #push_to_hub=True,
)

# tokenized_train = train_ds.map(encode, batched=True)
# tokenized_valid = valid_ds.map(encode, batched=True)

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds.select(range(10)),
    eval_dataset=valid_ds.select(range(10)),
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
print("Answer: ", decoded_answer)
