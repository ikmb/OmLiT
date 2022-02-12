# O-Link

A Rust library with a python binder to it. The OLink library is a library used for preparing and encoding multi-omics data for training, retraining, and/or running inferences on peptide HLA-II interactions.

## Aim

The aim of the library is provide a state of the art preprocessing library for either training the models or for running inference.

The library support the following encoding tasks:

1. Sequence only models, where the major aim is to train models using sequence only information, the library support two types of input data: quantitative data and immunopeptidomics data. Incase of quantitative data, the dataset can be used AS IS with out further editing, meanwhile, immunopeptidomics data only contain positive peptides and hence we must generate negative peptides. The library support 4 different methods to generate negative examples:
    1. Shuffling, where the positive examples are shuffled to generate negative examples.
    2. Proteome-sampling, where negative examples are randomly sampled from the proteome.
    3. Expressed-genes sampling, where the sampling is from a specific tissues, i.e. sampling will be from the same genes where it is expressed.
    4. same protein, where sampling, i.e. negatives sampling, is from the same protein as the positives, tradeoff for using each method are discussed in  ElAbd et al paper (In Preparation,).

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
    1. Get the index of the target GO term in the library constant array*.
    2. If term is not located return the default index 1048.
    3. Update the the position pointed to by the index to 1.
3. Return the encoded array.

## building it from the source

1. install Rust on your machine from the main website or using (skip this function if you have Rust already) 

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. clone the directory locally

```bash
git clone https://github.com/ikmb/O-Link-
```

3. Create a new conda environment and active it

```bash
conda create -n o_link -y && conda activate o_link
```

4. install maturin

```bash
pip install maturin 
```

5. build the code

```bash
cd O-Link- && maturin develop --release 
```

6. Load the library into Python and start working

```python

from OLink import * # load all the function // otherwise we can use
import OLink as linker # e.g. to have have a clean name space
?generate_train_based_on_seq_exp_subcell # print help message about a specific functions 
```

## using the library !!

The python scripts defined at the test and the benchmarking directories shall be used as a good start for learning Rust.

## Funding

The project was funded by the German Research Foundation (DFG) (Research Training Group 1743, ‘Genes, Environment and Inflammation’)
