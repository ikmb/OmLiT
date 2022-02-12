#!/usr/bin/env python 
"""
@author: Hesham ElAbd
@contact: h.elabd@ikmb.uni-kiel.de
@brief: Testing the code of the O-Link library for the correctness of generation
"""
## Load the modules
#------------------
from tqdm import tqdm 
import OLink as linker
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
## Define some constants
#-----------------------
SIZE_OF_DB=[1,10,100,1000,10_000,100_000,1000_000,1000_000_0,1000_000_00]
encode_time=[]
tool_name=[]
database_size=[]
# 1. Benchmarking the library encoder against TensorFlow encoder
#--------------------------------------------------------------
print(f"{time.ctime()} INFO:: Checking the correctness of the encode ....")
for size in tqdm(SIZE_OF_DB):
    ## Load the database 
    database=["TESTPEPTIDE"]*size
    ## recording the time of the encoder 
    start_time=time.time()
    encoded_test=linker.encode_sequence(database,21)
    end_time=time.time()
    encode_time.append(end_time-start_time)
    tool_name.append('O-LINK')
    database_size.append(size)
    ## record the tensorflow tokenization+encoding
    start_time=time.time()
    tokenizer_new=tf.keras.preprocessing.text.Tokenizer(num_words=26,char_level=True)
    tokenizer_new.fit_on_sequences(database)    
    encoded_text= tf.keras.preprocessing.sequence.pad_sequences(sequences=
        tokenizer_new.texts_to_sequences(database),dtype=np.uint8,
        maxlen=21,padding="pre")
    end_time=time.time()
    encode_time.append(end_time-start_time)
    tool_name.append('TensorFlow (Tokenizer+Encode)')
    database_size.append(size)
    ## record the tensorflow encoding
    start_time=time.time()
    encoded_text= tf.keras.preprocessing.sequence.pad_sequences(sequences=
        tokenizer_new.texts_to_sequences(database),dtype=np.uint8,
        maxlen=21,padding="pre")
    end_time=time.time()
    encode_time.append(end_time-start_time)
    tool_name.append('TensorFlow (Encode)')
    database_size.append(size)

## Define a data frame to store the results
#-------------------------------------------
input_table=pd.DataFrame({
    'input_size':database_size,
    'time':encode_time,
    'Tool Name':tool_name
})

## Plot the results
#------------------
plt.style.use('seaborn-whitegrid')
ax=sns.lineplot(data=input_table,x='input_size',y='time',hue='Tool Name')
ax.set_xlabel('Number of peptides')
ax.set_ylabel('Encoding time (seconds)')
ax.set_title('Execution time of OLink Vs. TensorFlow',loc='left')
plt.savefig('execution_time_O_LINK_vs_TF.png',dpi=1200)