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
FOLD_NEG=5
THRESHOLD=1
positive_peptides=dict()
for seq in SeqIO.parse('/Users/heshamelabd/projects/RustDB/datasets/human_proteome_pos.fasta','fasta'):
    positive_peptides[seq.id]=str(seq.seq)

input=([])

linker.sample_negatives_from_positives_no_test((['SGASGPENFQVG']*10_000,['DRB1_1501;DRB_1301;1506']*10_000,['total PBMC']*10_000),
positive_peptides,PATH2CASHED_DB,FOLD_NEG,THRESHOLD)



