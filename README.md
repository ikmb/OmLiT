# O-Link

A Rust library with a python binder to it. The OLink library is a library used for preparing and encoding multi-omics data for training, retraining, and/or running inferences on peptide HLA-II interactions.

## Aim

The aim of the library is provide a state of the art preprocessing library for either training the models or for running inference.

The library support the following encoding tasks:

1. Sequence only models, where the major aim is to train models using sequence only information, the library support two types of input data: quantitative data and immunopeptidomics data. Incase of quantitative data, the dataset can be used AS IS with out further editing, meanwhile, immunopeptidomics data only contain positive peptides and hence we must generate negative peptides. The library support 4 different methods to generate negative examples:
    i. Shuffling, where the positive examples are shuffled to generate negative examples.
    ii. Proteome-sampling, where negative examples are randomly sampled from the proteome.
    iii. Expressed-genes sampling, where the sampling is from a specific tissues, i.e. sampling will be from the same genes where it is expressed.
    iv. same protein, where sampling, i.e. negatives sampling, is from the same protein as the positives, tradeoff for using each method are discussed in  ElAbd et al paper (In Preparation,).

2. Multi-omics, here it means data generated from different Omics layer beside the sequence information, the following information are used
    i. expression data, where the expression level of the parent transcript for the positive and the negative peptides are included.
    ii. subcellular locations, where the subcellular location of every protein is considered
    iv. distance to glycosylation, where the distance to the nearest glycosylation site is considered

Incase of different omics data, the models will be trained on different combination of omics, e.g. sequence information and expression, or on sequence information, expression and distance to glycosylation. see below for more details.

## Encoding algorithms

The sampling algorithms generate a dataset of positives and negatives that machine-learning can not use directly as they are not numerically encoded. Hence, numerical encoding is a calculation heavy prerequisite step, the encoding algorithms are explained below.

### Sequence Encoding

Hardcoded build-in scheme for the numerical translation of amino acids into digits were:

1. Shorter sequence are zero padded into a fixed maximum length , default is 21.
2. Longer sequences are trimmed into shorter sequences, default is 21.
3. Amino acids are translated into vectors of u8 digits, e.g. TKLO…L =>[1,13,16,4,…,8].
4. Executed on parallel and returns a NumPy arrays with shape (num_peptides, max_length).

### Encoding sub cellular locations

Using gene ontology database with terms quantified with ‘located_in’. The encoding is done using ‘multi-hot’ arrays because proteins can be allocated in different cellular compartments.
The encoding is made of the following elements

1. Allocate a vector of zeros with shape of [1,1049]
2. For location in locations:
    i. Get the index of the target GO term in the library constant array*.
    ii. If term is not located return the default index 1048.
    iii. Update the the position pointed to by the index to 1.
3. Return the encoded array.

## building it from the source



## Usage


## Performance
