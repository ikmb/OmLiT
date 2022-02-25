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
alleles=['DRB1_1101']*len(development_seq)
tissue=['total PBMC']*len(development_seq)
INPUT=(development_seq,alleles,tissue)

positive_peptides=dict()
for seq in SeqIO.parse('/Users/heshamelabd/projects/RustDB/datasets/human_proteome_pos.fasta','fasta'):
    positive_peptides[seq.id]=str(seq.seq)

## Testing the code
#------------------
print(f"Testing the dataset with a model train on sequence and expression only ..., starting at: {time.ctime()}")
try:
    (train_data, test_data, unmapped_train, unmapped_test)=linker.generate_train_based_on_seq_exp(INPUT,
                positive_peptides,PATH2CASHED_DB,PSEUDO_SEQ,MAX_LEN,1,10,
                TEST_SIZE)
    print('Testing with random shuffling passed, ...')
except Exception as exp:
    print(f"Test failed ... {str(exp)}")
assert len(train_data)==4, "function retrained in correct number of elements for the train array"
assert len(test_data)==4, "function retrained in correct number of elements for the test array"

print(f"Testing the dataset with a model train on sequence, expression and sub-cellular location ..., starting at: {time.ctime()}")
try:
    (train_data, test_data, unmapped_train, unmapped_test)=linker.generate_train_based_on_seq_exp_subcell(INPUT,
                positive_peptides,PATH2CASHED_DB,PSEUDO_SEQ,MAX_LEN,1,10,
                TEST_SIZE)
    print('Testing with random shuffling passed, ...')
except Exception as exp:
    print(f"Test failed ... {str(exp)}")
assert len(train_data)==5, "function retrained in correct number of elements for the train array"
assert len(test_data)==5, "function retrained in correct number of elements for the test array"

print(f"Testing the dataset with a model train on sequence, expression, sub-cellular location and context ..., starting at: {time.ctime()}")
try:
    (train_data, test_data, unmapped_train, unmapped_test)=linker.generate_train_based_on_seq_exp_subcell_context(INPUT,
                positive_peptides,PATH2CASHED_DB,PSEUDO_SEQ,MAX_LEN,1,10,
                TEST_SIZE)
    print('Testing with random shuffling passed, ...')
except Exception as exp:
    print(f"Test failed ... {str(exp)}")
assert len(train_data)==6, "function retrained in correct number of elements for the train array"
assert len(test_data)==6, "function retrained in correct number of elements for the test array"

print(f"Testing the dataset with a model train on sequence, expression, sub-cellular location, context and distance to glycosylation ..., starting at: {time.ctime()}")
try:
    (train_data, test_data, unmapped_train, unmapped_test)=linker.generate_train_based_on_seq_exp_subcell_loc_context_d2g(INPUT,
                positive_peptides,PATH2CASHED_DB,PSEUDO_SEQ,MAX_LEN,1,10,
                TEST_SIZE)
    print('Testing with random shuffling passed, ...')
except Exception as exp:
    print(f"Test failed ... {str(exp)}")
assert len(train_data)==7, "function retrained in correct number of elements for the train array"
assert len(test_data)==7, "function retrained in correct number of elements for the test array"