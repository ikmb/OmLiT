#!/usr/bin/env python 
"""
@author: Hesham ElAbd
@contact: h.elabd@ikmb.uni-kiel.de
@brief: Testing the code of the O-Link library for the correctness of generation training dataset from a set of positive peptides 
"""
## Load the models
#-----------------
import OmLiT as linker
import pandas as pd 
import numpy as np 
from Bio import SeqIO 

## Define some error messages 
ERROR_ONE="IN CORRECT OUTPUT SHAPE"
ERROR_TWO="IN CORRECT SUM OF ELEMENTS"
ERROR_THREE="CRITICAL ERROR WAS ENCOUNTERERS"
ERROR_FOUR="Train and test dataset are over lapping"
## Load the datasets 
#-------------------
development_seq=list(set(pd.read_csv('/Users/heshamelabd/projects/RustDB/datasets/dev_peptides.txt').iloc[:,0].to_list())) # data will be uploaded upon the acceptance of the manuscript
positive_peptides=dict()
for seq in SeqIO.parse('/Users/heshamelabd/projects/RustDB/datasets/human_proteome_pos.fasta','fasta'):
    positive_peptides[seq.id]=str(seq.seq)
## generate a database through shuffling
#---------------------------------------
try: 
    ((train_dataset),(test_dataset))=linker.generate_train_ds_shuffling_sm(
        development_seq,10,21,0.2)
    print(f"Test one passes, ..., the function did not panic!! while calling it")
except: 
    raise RuntimeError(ERROR_THREE)

assert train_dataset[0].shape[1]==21, ERROR_ONE
assert test_dataset[0].shape[1]==21, ERROR_ONE
assert train_dataset[0].shape[0]==train_dataset[1].shape[0], ERROR_ONE
assert test_dataset[0].shape[0]==test_dataset[1].shape[0], ERROR_ONE


## Check that there not overlap in the sequences between test and train and between positives 
#---------------------------------------------------------------------------------------------
train_seq=set([''.join([str(elem) for elem in train_dataset[0][idx,:]]) for idx in range(train_dataset[0].shape[0])])
test_seq=set([''.join([str(elem) for elem in test_dataset[0][idx,:]]) for idx in range(test_dataset[0].shape[0])])
assert len(train_seq.intersection(test_seq))==0,ERROR_FOUR

## Assert that positive and negatives does not overlap for the train and test dataset
#---------------------
positive_index_train=np.asarray(train_dataset[1].reshape(-1) == 1).nonzero()[0]
train_positive=set([''.join([str(elem) for elem in train_dataset[0][idx,:]]) for idx in positive_index_train])
train_negative=set([''.join([str(elem) for elem in train_dataset[0][idx,:]]) for idx in range(positive_index_train[-1]+1,train_dataset[0].shape[0],1)])
assert len(train_positive.intersection(train_negative))==0,ERROR_FOUR

positive_index_test=np.asarray(test_dataset[1].reshape(-1) == 1).nonzero()[0]
train_positive=set([''.join([str(elem) for elem in test_dataset[0][idx,:]]) for idx in positive_index_test])
train_negative=set([''.join([str(elem) for elem in test_dataset[0][idx,:]]) for idx in range(positive_index_test[-1]+1,test_dataset[0].shape[0],1)])
assert len(train_positive.intersection(train_negative))==0,ERROR_FOUR

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##

## generate a database through same protein sampling 
#---------------------------------------------------
try: 
    ((train_dataset),(test_dataset))=linker.generate_train_ds_proteome_sampling_sm(
        development_seq,positive_peptides,10,21,0.2)
    print(f"Test one Two, ..., the function did not panic!! while calling it")
except: 
    raise RuntimeError(ERROR_THREE)

assert train_dataset[0].shape[1]==21, ERROR_ONE
assert test_dataset[0].shape[1]==21, ERROR_ONE
assert train_dataset[0].shape[0]==train_dataset[1].shape[0], ERROR_ONE
assert test_dataset[0].shape[0]==test_dataset[1].shape[0], ERROR_ONE


## Check that there not overlap in the sequences between test and train and between positives 
#---------------------------------------------------------------------------------------------
train_seq=set([''.join([str(elem) for elem in train_dataset[0][idx,:]]) for idx in range(train_dataset[0].shape[0])])
test_seq=set([''.join([str(elem) for elem in test_dataset[0][idx,:]]) for idx in range(test_dataset[0].shape[0])])
assert len(train_seq.intersection(test_seq))==0,ERROR_FOUR

## Assert that positive and negatives does not overlap for the train and test dataset
#---------------------
positive_index_train=np.asarray(train_dataset[1].reshape(-1) == 1).nonzero()[0]
train_positive=set([''.join([str(elem) for elem in train_dataset[0][idx,:]]) for idx in positive_index_train])
train_negative=set([''.join([str(elem) for elem in train_dataset[0][idx,:]]) for idx in range(positive_index_train[-1]+1,train_dataset[0].shape[0],1)])
assert len(train_positive.intersection(train_negative))==0,ERROR_FOUR

positive_index_test=np.asarray(test_dataset[1].reshape(-1) == 1).nonzero()[0]
train_positive=set([''.join([str(elem) for elem in test_dataset[0][idx,:]]) for idx in positive_index_test])
train_negative=set([''.join([str(elem) for elem in test_dataset[0][idx,:]]) for idx in range(positive_index_test[-1]+1,test_dataset[0].shape[0],1)])
assert len(train_positive.intersection(train_negative))==0,ERROR_FOUR

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##

## generate a database through same protein sampling 
#---------------------------------------------------
try: 
    ((train_dataset),(test_dataset))=linker.generate_train_ds_same_protein_sampling_sm(
        development_seq,positive_peptides,10,21,0.2)
    print(f"Test one Three, ..., the function did not panic!! while calling it")
except: 
    raise RuntimeError(ERROR_THREE)

assert train_dataset[0].shape[1]==21, ERROR_ONE
assert test_dataset[0].shape[1]==21, ERROR_ONE
assert train_dataset[0].shape[0]==train_dataset[1].shape[0], ERROR_ONE
assert test_dataset[0].shape[0]==test_dataset[1].shape[0], ERROR_ONE


## Check that there not overlap in the sequences between test and train and between positives 
#---------------------------------------------------------------------------------------------
train_seq=set([''.join([str(elem) for elem in train_dataset[0][idx,:]]) for idx in range(train_dataset[0].shape[0])])
test_seq=set([''.join([str(elem) for elem in test_dataset[0][idx,:]]) for idx in range(test_dataset[0].shape[0])])
assert len(train_seq.intersection(test_seq))==0,ERROR_FOUR

## Assert that positive and negatives does not overlap for the train and test dataset
#---------------------
positive_index_train=np.asarray(train_dataset[1].reshape(-1) == 1).nonzero()[0]
train_positive=set([''.join([str(elem) for elem in train_dataset[0][idx,:]]) for idx in positive_index_train])
train_negative=set([''.join([str(elem) for elem in train_dataset[0][idx,:]]) for idx in range(positive_index_train[-1]+1,train_dataset[0].shape[0],1)])
assert len(train_positive.intersection(train_negative))==0,ERROR_FOUR

positive_index_test=np.asarray(test_dataset[1].reshape(-1) == 1).nonzero()[0]
train_positive=set([''.join([str(elem) for elem in test_dataset[0][idx,:]]) for idx in positive_index_test])
train_negative=set([''.join([str(elem) for elem in test_dataset[0][idx,:]]) for idx in range(positive_index_test[-1]+1,test_dataset[0].shape[0],1)])
assert len(train_positive.intersection(train_negative))==0,ERROR_FOUR


##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++##

## generate a database through same protein sampling 
#---------------------------------------------------
try: 
    ((train_dataset),(test_dataset))=linker.generate_train_ds_expressed_protein_sampling_sm(
        development_seq,positive_peptides,'/Users/heshamelabd/projects/RustDB/datasets/cashed_database.db',
        'total PBMC',1,10,21,0.2)
    print(f"Test one Four, ..., the function did not panic!! while calling it")
except: 
    raise RuntimeError(ERROR_THREE)

assert train_dataset[0].shape[1]==21, ERROR_ONE
assert test_dataset[0].shape[1]==21, ERROR_ONE
assert train_dataset[0].shape[0]==train_dataset[1].shape[0], ERROR_ONE
assert test_dataset[0].shape[0]==test_dataset[1].shape[0], ERROR_ONE


## Check that there not overlap in the sequences between test and train and between positives 
#---------------------------------------------------------------------------------------------
train_seq=set([''.join([str(elem) for elem in train_dataset[0][idx,:]]) for idx in range(train_dataset[0].shape[0])])
test_seq=set([''.join([str(elem) for elem in test_dataset[0][idx,:]]) for idx in range(test_dataset[0].shape[0])])
assert len(train_seq.intersection(test_seq))==0,ERROR_FOUR

## Assert that positive and negatives does not overlap for the train and test dataset
#---------------------
positive_index_train=np.asarray(train_dataset[1].reshape(-1) == 1).nonzero()[0]
train_positive=set([''.join([str(elem) for elem in train_dataset[0][idx,:]]) for idx in positive_index_train])
train_negative=set([''.join([str(elem) for elem in train_dataset[0][idx,:]]) for idx in range(positive_index_train[-1]+1,train_dataset[0].shape[0],1)])
assert len(train_positive.intersection(train_negative))==0,ERROR_FOUR

positive_index_test=np.asarray(test_dataset[1].reshape(-1) == 1).nonzero()[0]
train_positive=set([''.join([str(elem) for elem in test_dataset[0][idx,:]]) for idx in positive_index_test])
train_negative=set([''.join([str(elem) for elem in test_dataset[0][idx,:]]) for idx in range(positive_index_test[-1]+1,test_dataset[0].shape[0],1)])
assert len(train_positive.intersection(train_negative))==0,ERROR_FOUR