#!/usr/bin/env python 
"""
@author: Hesham ElAbd 
@brief: analyzing the performance of the aspects of the library
    1- Reading and parsing Quantitative data against the models 
    2- generating data from multi-modal pure sequence models 
"""
## Loading the modules 
#---------------------
import time 
import OmLiT as linker 
import pandas as pd 
import numpy as np 
from Bio import SeqIO

## Define some constants
#-----------------------
PSEUDO_SEQ='/Users/heshamelabd/projects/RustDB/datasets/pseudosequence.2016.all.X.dat'
PATH2CASHED_DB='/Users/heshamelabd/projects/RustDB/datasets/cashed_database.db'
TEST_SIZE=0.4
MAX_LEN=21

## Load the datasets
#-------------------
development_seq=list(set(pd.read_csv('/Users/heshamelabd/projects/RustDB/datasets/dev_peptides.txt').iloc[:,0].to_list())) # data will be uploaded upon the acceptance of the manuscript
simulated_data={
    'DRB1_1501':development_seq,
    'DRB1_1301':development_seq,
    'DRB1_0301':development_seq,
    'DRB1_0103':development_seq
}
positive_peptides=dict()
for seq in SeqIO.parse('/Users/heshamelabd/projects/RustDB/datasets/human_proteome_pos.fasta','fasta'):
    positive_peptides[seq.id]=str(seq.seq)

## Testing the code
#------------------
print(f"Testing the dataset with shuffling ..., starting at: {time.ctime()}")
try:
    (train_data, test_data)=linker.generate_train_ds_pm(simulated_data,positive_peptides,None,None,PSEUDO_SEQ,10,None,MAX_LEN,TEST_SIZE,'shuffle')
    print('Testing with random shuffling passed, ...')
except:
    print("Test failed ...")

print(f"Testing the dataset with random sampling from the proteome ..., starting at: {time.ctime()}")
try:
    (train_data, test_data)=linker.generate_train_ds_pm(simulated_data,positive_peptides,PATH2CASHED_DB,
                'total PBMC',PSEUDO_SEQ,10,1,MAX_LEN,TEST_SIZE,'prot_sample')
    print('Testing with proteome sampling passed, ...')
except:
    print("Test failed ...") 

print(f"Testing the dataset with same protein sampling ..., starting at: {time.ctime()}")
try:
    (train_data, test_data)=linker.generate_train_ds_pm(simulated_data,positive_peptides,PATH2CASHED_DB,
                'total PBMC',PSEUDO_SEQ,10,1,MAX_LEN,TEST_SIZE,'prot_sample')
    print('Testing with proteome sampling passed, ...')
except:
    print("Test failed ...")

print(f"Testing the dataset with expressed proteome sampling ..., starting at: {time.ctime()}")
try:
    (train_data, test_data)=linker.generate_train_ds_pm(simulated_data,positive_peptides,PATH2CASHED_DB,
                'total PBMC',PSEUDO_SEQ,10,1,MAX_LEN,TEST_SIZE,'prot_sample')
    print('Testing with proteome sampling passed, ...')
except:
    print("Test failed ...") 