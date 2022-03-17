#!/usr/bin/env python 
"""
@author: Hesham ElAbd
@brief: testing the proteome sampling from different individuals 
"""
## Load the models
#----------------
import OmLiT as linker 
from Bio import SeqIO

## Define the test variables 
#---------------------------
PATH2CASHED_DB='/Users/heshamelabd/projects/RustDB/datasets/cashed_database.db'
PSEUDO_SEQ='/Users/heshamelabd/projects/RustDB/datasets/pseudosequence.2016.all.X.dat'
FOLD_NEG=5
THRESHOLD=1
positive_peptides=dict()
for seq in SeqIO.parse('/Users/heshamelabd/projects/RustDB/datasets/human_proteome_pos.fasta','fasta'):
    positive_peptides[seq.id]=str(seq.seq)

try: 
    results=linker.sample_negatives_from_positives_no_test((['SGASGPENFQVG']*1000,['DRB1_1501']*1000,['total PBMC']*1000),
positive_peptides,PATH2CASHED_DB,FOLD_NEG,THRESHOLD)
except Exception as exp:
    raise RuntimeError(f"Calling the function failed due to the following error: {exp}")

try:
    (encoded_results,unmapped_results)=linker.annotate_and_encode_input_sequences(results,21,positive_peptides,PATH2CASHED_DB,PSEUDO_SEQ,True)
except Exception as exp:
    raise RuntimeError(f"Calling the function failed due to the following error: {exp}")