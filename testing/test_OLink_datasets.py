#!/usr/bin/env python 
"""
@author: Hesham ElAbd 
@brief: analyzing the performance of the aspects of the library
    1- Reading and parsing Quantitative data against the models 
    2- generating data from multi-modal pure sequence models 
"""
## Loading the modules 
#---------------------
import OmLiT as linker 
import pandas as pd 
import numpy as np 

## Define some constants
#-----------------------
PSEUDO_SEQ='/Users/heshamelabd/projects/RustDB/datasets/pseudosequence.2016.all.X.dat'
INPUT_TABLE='/Users/heshamelabd/projects/RustDB/datasets/BA_database.tsv'
TEST_SIZE=0.2
MAX_LEN=21



## Test the reading and parsing of the Q models 
#----------------------------------------------
try:
    (train_database,test_database)=linker.generate_train_ds_Qd(INPUT_TABLE,PSEUDO_SEQ,TEST_SIZE,MAX_LEN)
    print("Test one passed, the function did not panic")
except:
    print("Test one failed, the function paniced")

## checking the correctness of the results
#-----------------------------------------
assert len(train_database)==3, f"incorrect number of dimension, expected 3, found: {len(train_database)}"
assert len(test_database)==3, f"incorrect number of dimension, expected 3, found: {len(train_database)}"

assert train_database[0].shape[0]+test_database[0].shape[0]==131_008, f"In correct number of examples, expected 131,008 peptides, instead found: {train_database[0].shape[0]+test_database[0].shape[0]}"
assert train_database[0].shape[1]==21, f"MisMatched padded peptide length, expected length to be 21, however, it is: {train_database[0].shape[1]}"
assert test_database[0].shape[1]==21, f"MisMatched padded peptide length, expected length to be 21, however, it is: {test_database[0].shape[1]}"
assert train_database[1].shape[1]==34, f"MisMatched padded pseudo sequence length, expected length to be 34, however, it is: {train_database[1].shape[1]}"
assert test_database[1].shape[1]==34, f"MisMatched padded pseudo sequence length, expected length to be 34, however, it is: {test_database[1].shape[1]}"



