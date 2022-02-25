#!/usr/bin/env python 
"""
@author: Hesham ElAbd
@brief: Loading the annotation database  
"""
## Load the datasets
#-------------------
import time 
import random 
from tqdm import tqdm 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec 
import OmLiT as linker 
from Bio import SeqIO
## loading and recording the time of different datasets 
#------------------------------------------------------
NUM_ELEMENTS=[64,128,256,512,1024,2048]
METHODS=['shuffling','proteome_sampling','expressed_protein_sampling','same_protein_sampling']
CASHED_DB='/Users/heshamelabd/projects/RustDB/datasets/cashed_database.db'
INPUT_DATASET=pd.read_csv('/Users/heshamelabd/projects/RustDB/benchmarking/In_house_E_1501_pure_datasets.txt',sep='\t',header=None).iloc[:,0].to_list()
positive_peptides=dict()
for seq in SeqIO.parse('/Users/heshamelabd/projects/RustDB/datasets/human_proteome_pos.fasta','fasta'):
    positive_peptides[seq.id]=str(seq.seq)

TISSUE_NAME='total PBMC'
NUM_TRIALS=3
FOLD_NEG=5
##----------------------
methods_cont,run_time_cont,trial_cont, input_size=[],[],[],[]
for num_element in tqdm(NUM_ELEMENTS):
    database=INPUT_DATASET[:num_element]
    for method in tqdm(METHODS):
        print(f"{time.ctime()} INFO:: BenchMarking the performance using method: {method} and {num_element} peptides")
        for trial in range(NUM_TRIALS):
            if method=='shuffling':
                start=time.time()
                res=linker.generate_train_ds_shuffling_sm(database, FOLD_NEG, 21, 0.3)
                end=time.time()
            elif method=='proteome_sampling':
                start=time.time()
                res=linker.generate_train_ds_proteome_sampling_sm(database, positive_peptides, FOLD_NEG, 21, 0.3)
                end=time.time()
            elif method=='expressed_protein_sampling': 
                start=time.time()
                res=linker.generate_train_ds_expressed_protein_sampling_sm(
                     database,positive_peptides,CASHED_DB,
                    TISSUE_NAME,1,FOLD_NEG,21,0.3)
                end=time.time()
            else: 
                start=time.time()
                res=linker.generate_train_ds_same_protein_sampling_sm(
                    database,positive_peptides,FOLD_NEG,21,0.3)
                end=time.time()
            ## store the results 
            #---------------------------------
            methods_cont.append(method)
            trial_cont.append(trial+1)
            run_time_cont.append(end-start)
            input_size.append(num_element)
#-------------------+++++++++++++++++++++++++++++++++
results_different_methods=pd.DataFrame({
    'input_size':input_size,
    'Method':methods_cont,
    'trial_index':trial_cont,
    'RunTime':run_time_cont
})
results_different_methods['log_input_size']=[np.log2(num) for num in results_different_methods['input_size'] ]
#-------------------+++++++++++++++++++++++++++++++++
## Test the impact of the encoding algorthism
#--------------------------------------------
### First encoding with different number of alleles
PSEUDO_SEQ='/Users/heshamelabd/projects/RustDB/datasets/pseudosequence.2016.all.X.dat'
PATH2CASHED_DB='/Users/heshamelabd/projects/RustDB/datasets/cashed_database.db'
TEST_SIZE=0.4
MAX_LEN=21

# Define the last layer 
#------------------------|
## Define the omics layer|
#------------------------|

# Define the omics from which the analysis will be conducted
#-----------------------------------------------------------
OMICS_LAYER=[
    'Seq+Txp',
    'Seq+Txp+SC',
    'Seq+Txp+SC+Context',
    'Seq+Txp+SC+Context+D2G',
]

omics_layer_name, run_time_cont, trial_cont, input_size=[],[],[],[]
for num_element in tqdm(NUM_ELEMENTS):
    peptides=INPUT_DATASET[:num_element]
    allele=['DRB1_1501']*len(peptides)
    tissues=['total PBMC']*len(peptides)
    database=(peptides,allele,tissues)
    for level in tqdm(OMICS_LAYER):
        print(f"{time.ctime()} INFO:: BenchMarking the performance using the Omics level: {level} and {num_element} peptides")
        for trial in range(NUM_TRIALS):
            if method=='Seq+Txp':
                start=time.time()
                (train_data, test_data, unmapped_train, unmapped_test)=linker.generate_train_based_on_seq_exp(database,
                        positive_peptides,PATH2CASHED_DB,PSEUDO_SEQ,MAX_LEN,1,FOLD_NEG,TEST_SIZE)
                end=time.time()
            elif method=='Seq+Txp+SC':
                start=time.time()
                (train_data, test_data, unmapped_train, unmapped_test)=linker.generate_train_based_on_seq_exp_subcell(database,
                        positive_peptides,PATH2CASHED_DB,PSEUDO_SEQ,MAX_LEN,1,FOLD_NEG,TEST_SIZE)
                end=time.time()
            elif method=='Seq+Txp+SC+Context':
                start=time.time()
                (train_data, test_data, unmapped_train, unmapped_test)=linker.generate_train_based_on_seq_exp_subcell_context(database,
                        positive_peptides,PATH2CASHED_DB,PSEUDO_SEQ,MAX_LEN,1,FOLD_NEG,TEST_SIZE)
                end=time.time()
            else: 
                start=time.time()
                (train_data, test_data, unmapped_train, unmapped_test)=linker.generate_train_based_on_seq_exp_subcell_loc_context_d2g(database,
                        positive_peptides,PATH2CASHED_DB,PSEUDO_SEQ,MAX_LEN,1,FOLD_NEG,TEST_SIZE)
                end=time.time()
            ## store the results 
            #---------------------------------
            omics_layer_name.append(level)
            trial_cont.append(trial+1)
            run_time_cont.append(end-start)
            input_size.append(num_element)
#-------------------+++++++++++++++++++++++++++++++++
results_different_omics_layer=pd.DataFrame({
    'input_size':input_size,
    'omics_level':omics_layer_name,
    'trial_index':trial_cont,
    'RunTime':run_time_cont
})
results_different_omics_layer['log_input_size']=[np.log2(num) for num in results_different_omics_layer['input_size']]

########### +++++++++++++++++++++++++++++++++++++++++++++++++++++++ ###########

TISSUES=['total PBMC','liver','colon','small intestine','kidney']
OMICS_LAYER=[
    'Seq+Txp',
    'Seq+Txp+SC',
    'Seq+Txp+SC+Context',
    'Seq+Txp+SC+Context+D2G',
]

omics_layer_name, run_time_cont, trial_cont, input_size=[],[],[],[]
for num_element in tqdm(NUM_ELEMENTS):
    peptides=INPUT_DATASET[:num_element]
    allele=['DRB1_1501']*len(peptides)
    tissues=[random.choice(TISSUES) for _ in range(len(peptides))]
    database=(peptides,allele,tissues)
    for level in tqdm(OMICS_LAYER):
        print(f"{time.ctime()} INFO:: BenchMarking the performance using the Omics level: {level} and {num_element} peptides")
        for trial in range(NUM_TRIALS):
            if method=='Seq+Txp':
                start=time.time()
                (train_data, test_data, unmapped_train, unmapped_test)=linker.generate_train_based_on_seq_exp(database,
                        positive_peptides,PATH2CASHED_DB,PSEUDO_SEQ,MAX_LEN,1,FOLD_NEG,TEST_SIZE)
                end=time.time()
            elif method=='Seq+Txp+SC':
                start=time.time()
                (train_data, test_data, unmapped_train, unmapped_test)=linker.generate_train_based_on_seq_exp_subcell(database,
                        positive_peptides,PATH2CASHED_DB,PSEUDO_SEQ,MAX_LEN,1,FOLD_NEG,TEST_SIZE)
                end=time.time()
            elif method=='Seq+Txp+SC+Context':
                start=time.time()
                (train_data, test_data, unmapped_train, unmapped_test)=linker.generate_train_based_on_seq_exp_subcell_context(database,
                        positive_peptides,PATH2CASHED_DB,PSEUDO_SEQ,MAX_LEN,1,FOLD_NEG,TEST_SIZE)
                end=time.time()
            else: 
                start=time.time()
                (train_data, test_data, unmapped_train, unmapped_test)=linker.generate_train_based_on_seq_exp_subcell_loc_context_d2g(database,
                        positive_peptides,PATH2CASHED_DB,PSEUDO_SEQ,MAX_LEN,1,FOLD_NEG,TEST_SIZE)
                end=time.time()
            ## store the results 
            #---------------------------------
            omics_layer_name.append(level)
            trial_cont.append(trial+1)
            run_time_cont.append(end-start)
            input_size.append(num_element)
#-------------------+++++++++++++++++++++++++++++++++
results_different_omics_layer_5_tissues=pd.DataFrame({
    'input_size':input_size,
    'omics_level':omics_layer_name,
    'trial_index':trial_cont,
    'RunTime':run_time_cont
})
results_different_omics_layer_5_tissues['log_input_size']=[np.log2(num) for num in results_different_omics_layer_5_tissues['input_size']]

#+++++++++++++++++++++
ALLELES=['DRB1_1501','DRB1_1301','DRB1_1101','DRB1_0103','DRB1_0301']
OMICS_LAYER=[
    'Seq+Txp',
    'Seq+Txp+SC',
    'Seq+Txp+SC+Context',
    'Seq+Txp+SC+Context+D2G',
]

omics_layer_name, run_time_cont, trial_cont, input_size=[],[],[],[]
for num_element in tqdm(NUM_ELEMENTS):
    peptides=INPUT_DATASET[:num_element]
    allele=[random.choice(ALLELES) for _ in range(len(peptides))]
    tissues=['total PBMC']*len(peptides)
    database=(peptides,allele,tissues)
    for level in tqdm(OMICS_LAYER):
        print(f"{time.ctime()} INFO:: BenchMarking the performance using the Omics level: {level} and {num_element} peptides")
        for trial in range(NUM_TRIALS):
            if method=='Seq+Txp':
                start=time.time()
                (train_data, test_data, unmapped_train, unmapped_test)=linker.generate_train_based_on_seq_exp(database,
                        positive_peptides,PATH2CASHED_DB,PSEUDO_SEQ,MAX_LEN,1,FOLD_NEG,TEST_SIZE)
                end=time.time()
            elif method=='Seq+Txp+SC':
                start=time.time()
                (train_data, test_data, unmapped_train, unmapped_test)=linker.generate_train_based_on_seq_exp_subcell(database,
                        positive_peptides,PATH2CASHED_DB,PSEUDO_SEQ,MAX_LEN,1,FOLD_NEG,TEST_SIZE)
                end=time.time()
            elif method=='Seq+Txp+SC+Context':
                start=time.time()
                (train_data, test_data, unmapped_train, unmapped_test)=linker.generate_train_based_on_seq_exp_subcell_context(database,
                        positive_peptides,PATH2CASHED_DB,PSEUDO_SEQ,MAX_LEN,1,FOLD_NEG,TEST_SIZE)
                end=time.time()
            else: 
                start=time.time()
                (train_data, test_data, unmapped_train, unmapped_test)=linker.generate_train_based_on_seq_exp_subcell_loc_context_d2g(database,
                        positive_peptides,PATH2CASHED_DB,PSEUDO_SEQ,MAX_LEN,1,FOLD_NEG,TEST_SIZE)
                end=time.time()
            ## store the results 
            #---------------------------------
            omics_layer_name.append(level)
            trial_cont.append(trial+1)
            run_time_cont.append(end-start)
            input_size.append(num_element)
#-------------------+++++++++++++++++++++++++++++++++
results_different_omics_layer_5_alleles=pd.DataFrame({
    'input_size':input_size,
    'omics_level':omics_layer_name,
    'trial_index':trial_cont,
    'RunTime':run_time_cont
})
results_different_omics_layer_5_alleles['log_input_size']=[np.log2(num) for num in results_different_omics_layer_5_alleles['input_size']]
#++++++++++++++
# making the figure to visualize the results
#-------------------------------------------
plt.style.use('seaborn-whitegrid')
fig = plt.figure()
fig.set_figwidth(19*(1/2.54))
fig.set_figheight(18*(1/2.54))
gs1 = gridspec.GridSpec(2, 4)
ax1=fig.add_subplot(gs1[0,:2])
ax2=fig.add_subplot(gs1[0,2:])
ax3=fig.add_subplot(gs1[1,:2])
ax4=fig.add_subplot(gs1[1,2:])
fig.tight_layout()
#----------------------------------------------
## Creating Figure 1A
#--------------------
sns.lineplot(data=pd.read_csv('/Users/heshamelabd/projects/RustDB/benchmarking/benchmarking_encoder.tsv',sep='\t'),
    x='input_size',y='time',hue='Tool Name',ax=ax1)
ax1.set_xlabel('Input size')
ax1.set_ylabel('Runtime (sec)')
ax1.set_title('A',loc='left')
##++++++++++++++
## Creating Figure 1B
#--------------------
sns.lineplot(data=pd.read_csv('/Users/heshamelabd/projects/RustDB/benchmarking/Benchmarking_single_model_preprocessing_performance.tsv',sep='\t'),
    x='log_input_size',y='RunTime',hue='Method',ax=ax2)
ax2.set_xlabel('Log2(Input size)')
ax2.set_ylabel('Runtime (sec)')
ax2.set_title('B',loc='left')
## Creating Figure 1C
#--------------------
sns.lineplot(data=results_different_omics_layer,
    x='log_input_size',y='RunTime',hue='omics_level',ax=ax3)
ax3.set_xlabel('Log2(Input size)')
ax3.set_ylabel('Runtime (sec)')
ax3.set_yticks(np.arange(0,120,10))
ax3.set_title('C',loc='left')
## Creating Figure 1D
#--------------------
sns.lineplot(data=results_different_omics_layer_5_alleles,
    x='log_input_size',y='RunTime',hue='omics_level',ax=ax4)
ax4.set_xlabel('Log2(Input size)')
ax4.set_ylabel('Runtime (sec)')
ax4.set_yticks(np.arange(0,120,10))
ax4.set_title('D',loc='left')
#++++++++++++++++++++++++++++
fig.tight_layout()