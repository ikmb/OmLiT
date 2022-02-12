#!/usr/bin/env python 
"""
@author: Hesham ElAbd
@contact: h.elabd@ikmb.uni-kiel.de
@brief: Testing the code of the O-Link library for the correctness of generation
"""
## Load the modules
#------------------
import OLink as linker
import numpy as np
import time

# define some constant values
#----------------------------
ERROR_ONE="IN CORRECT OUTPUT SHAPE"
ERROR_TWO="IN CORRECT SUM OF ELEMENTS"

# 1. testing the encoder 
#-----------------------
print(f"{time.ctime()} INFO:: Checking the correctness of the encode ....")



## encode input sequences
#------------------------
CASE_ONE=[""]
encoded_sequences=linker.encode_sequence(CASE_ONE,21)
assert encoded_sequences.shape==(1,21), ERROR_ONE
assert np.sum(encoded_sequences==0,axis=1)==21,ERROR_TWO

CASE_TWO=["TESTPEPTIDE"] 
encoded_sequences=linker.encode_sequence(CASE_TWO,21)
assert encoded_sequences.shape==(1,21), ERROR_ONE
assert np.sum(encoded_sequences==0,axis=1)==10,ERROR_TWO

CASE_THREE=["PEPTIDEONE","PEPTIDETWO","PEPTIDETHREE"]
encoded_sequences=linker.encode_sequence(CASE_THREE,21)
assert encoded_sequences.shape==(3,21), ERROR_ONE

