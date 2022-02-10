/// Load the libraries 
use pyo3::prelude::*; 
use std::collections::HashMap;
use ndarray::Dim;
use numpy::{PyArray, ToPyArray};
use pyo3::Python;
use crate::{utils::*, omics_builder::*};

/// declare and implement the functions 
///------------------------------------
/// ### Signature
/// generate_train_based_on_seq_exp(input2prepare:Tuple[List[str], List[str], List[str]], 
///             protoem:Dict[str,str], path2cashed_database:str,
///             pseudo_sequence:str, max_len:int, threshold:float, 
///             fold_neg:int, test_size:float)->Tuple[
///             Tuple[np.ndarray[shape=(Num_mapped_peptides,max_len),type=np.uint8],np.ndarray[shape=(Num_mapped_peptides,34),type=np.uint8]
///                   np.ndarray[shape=(Num_mapped_peptides,1),type=np.float32], np.ndarray[shape=(Num_mapped_peptides,1),type=np.uint8]
///                 ],
///             Tuple[Tuple[np.ndarray[shape=(Num_mapped_peptides,max_len),type=np.uint8],np.ndarray[shape=(Num_mapped_peptides,34),type=np.uint8]
///                   np.ndarray[shape=(Num_mapped_peptides,1),type=np.float32], np.ndarray[shape=(Num_mapped_peptides,1),type=np.uint8]
///                 ],
///             Tuple[List[str],List[str],List[str],List[str]],
///             Tuple[List[str],List[str],List[str],List[str]] 
///     ]
/// 
/// ### Summary
/// A python binder to the omics_builder module written in Rust, enabling efficient and parallel processing and preparation of the input data. 
/// 
/// ### Parameters  
/// input2prepare (Tuple[List[str], List[str], List[str]]): A tuple representing the main inputs to the code, this tuple is composite of three elements 
///     1. a list of strings representing input peptides 
///     2. a list of strings representing allele names
///     3. a list of strings representing the name of tissues from which the peptide was observed.
/// NOTE: all inputs must have the same length, otherwise Rust code will panic. 
/// 
/// protoem (Dict[str,str]): A dict representing the input to the function which is composite of protein name (uniprot-id) and its sequence.  
/// 
/// path2cashed_database (str): The path to load the cashed database which is a precomputed and cashed data structure used for annotating input proteins. 
///     See online documentation for more details.  
/// 
/// pseudo_sequence (str): The path to load the pseudo-sequences which is table containing the allele name and its pseudo-sequence.
/// 
/// max_len (int): The maximum length of of the peptide, longer sequences are trimmed and shorter sequences are zero-padded. 
/// 
/// threshold (float): A threshold used for generating the set of negative protein for each tissue, for more information about the meaning of this parameter check the
/// online documentation, specifically, the rust function (create_negative_database_from_positive which is defined in the utils module)
/// 
///  fold_neg (int): The amount of negative examples to sample, Xfold the training datasets, where 1 represent a ratio of 1 negative per each positive example while 5 represents  
///  5 negatives from each positive example.
/// 
/// test_size (int): The size of the test dataset, a float in the range (0,1) representing the fraction of the total amount of data 
/// 
/// ### Return
/// A tuple of 4 elements each with the following composition and structure: 
///     Tuple 1: Training tuple which is composite of 4 elements arranges as follow:
///         a. encoded peptide sequence --> a NumPy array of shape (num_mapped_train_peptides,max_len) and a type of np.uint8 representing the encoded peptide sequences   
///         b. encoded pseudo sequence  --> a NumPy array of shape (num_mapped_train_peptides,34) and a type of np.uint8 representing the encoded pseudo sequences   
///         c. encoded gene expression  --> a NumPy array of shape (num_mapped_train_peptides,1) and a type of np.float32 representing the gene expression of the parent transcript.  
///         d. encoded labels --> a NumPy of shape (num_mapped_train_peptides,1) and a type of np.uint8 representing the label of the peptide, where 1 represent binders, 0 represents non-binders
///     
///     Tuple 2: Testing tuples which is composite of 4 elements arranges as follow:
///         a. encoded peptide sequence --> a NumPy array of shape (num_mapped_test_peptides,max_len) and a type of np.uint8 representing the encoded peptide sequences   
///         b. encoded pseudo sequence  --> a NumPy array of shape (num_mapped_test_peptides,34) and a type of np.uint8 representing the encoded pseudo sequences   
///         c. encoded gene expression  --> a NumPy array of shape (num_mapped_test_peptides,1) and a type of np.float32 representing the gene expression of the parent transcript.  
///         d. encoded labels --> a NumPy of shape (num_mapped_test_peptides,1) and a type of np.uint8 representing the label of the peptide, where 1 represent binders, 0 represents non-binders
///     
///     Tuple 3: unmapped train data which is a tuple of four elements:
///         a. A Vector of strings representing the input peptide sequence. 
///         b. A Vector of strings representing the allele names. 
///         c. A Vector of strings representing the tissue names. 
///         d. A Vector of strings representing the peptide labels. 
/// 
///     Tuple 4: unmapped test data which is a tuple of four elements:
///         a. A Vector of strings representing the input peptide sequence. 
///         b. A Vector of strings representing the allele names. 
///         c. A Vector of strings representing the tissue names. 
///         d. A Vector of strings representing the peptide labels. 
#[pyfunction]
pub fn generate_train_based_on_seq_exp<'py>(py:Python<'py>,
        input2prepare:(Vec<String>,Vec<String>,Vec<String>),
                    proteome:HashMap<String,String>, path2cashed_db:String, 
                    path2pseudo_seq:String, max_len:usize,
        threshold:f32, fold_neg:u32, test_size:f32
    )->(
        /* Train Tensors */
        (&'py PyArray<u8,Dim<[usize;2]>> /*Train pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train encoded sequence */,
         &'py PyArray<f32,Dim<[usize;2]>> /*Train expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /* Train labels */), 
        
        /* Test Tensors */
        (&'py PyArray<u8,Dim<[usize;2]>> /*Test pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Test encoded sequence */,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /* Test labels */), 

        (Vec<String>,Vec<String>,Vec<String>, Vec<u8>),// unmapped train data points 
        (Vec<String>,Vec<String>,Vec<String>, Vec<u8>) // unmapped test data points
    )
{
    // Prepare the prelude for the encoders
    //------------------------------------- 
    let (train_data, test_data, pseudo_seq
            ,anno_table)=preparation_prelude(&input2prepare,&proteome,path2cashed_db,path2pseudo_seq,
            threshold,fold_neg,test_size);
    
    // encode and prepare the training data
    //-------------------------------------
    let (encoded_train_data,unmapped_train_data)=prepare_data_for_seq_and_expression_data(train_data,
        &pseudo_seq,max_len,&proteome,&anno_table); 
    
    let(encoded_test_data,unmapped_test_data)=prepare_data_for_seq_and_expression_data(test_data,
        &pseudo_seq,max_len,&proteome,&anno_table); 

    //return the results
    //------------------
    (
        (encoded_train_data.0.to_pyarray(py),encoded_train_data.1.to_pyarray(py),encoded_train_data.2.to_pyarray(py),encoded_train_data.3.to_pyarray(py)),
        (encoded_test_data.0.to_pyarray(py),encoded_test_data.1.to_pyarray(py),encoded_test_data.2.to_pyarray(py),encoded_test_data.3.to_pyarray(py)),
        unmapped_train_data,
        unmapped_test_data
    )
}
/// ### Signature
/// generate_train_based_on_seq_exp(input2prepare:Tuple[List[str], List[str], List[str]], 
///             protoem:Dict[str,str], path2cashed_database:str,
///             pseudo_sequence:str, max_len:int, threshold:float, 
///             fold_neg:int, test_size:float)->Tuple[
///             Tuple[np.ndarray[shape=(Num_mapped_peptides,max_len),type=np.uint8],np.ndarray[shape=(Num_mapped_peptides,34),type=np.uint8]
///                   np.ndarray[shape=(Num_mapped_peptides,1),type=np.float32], np.ndarray[shape=(Num_mapped_peptides,1),type=np.uint8]
///                 ],
///             Tuple[Tuple[np.ndarray[shape=(Num_mapped_peptides,max_len),type=np.uint8],np.ndarray[shape=(Num_mapped_peptides,34),type=np.uint8]
///                   np.ndarray[shape=(Num_mapped_peptides,1),type=np.float32], np.ndarray[shape=(Num_mapped_peptides,1),type=np.uint8]
///                 ],
///             Tuple[List[str],List[str],List[str],List[str]],
///             Tuple[List[str],List[str],List[str],List[str]] 
///     ]
/// 
/// ### Summary
/// A python binder to the omics_builder module written in Rust, enabling efficient and parallel processing and preparation of the input data. 
/// 
/// ### Parameters  
/// input2prepare (Tuple[List[str], List[str], List[str]]): A tuple representing the main inputs to the code, this tuple is composite of three elements 
///     1. a list of strings representing input peptides 
///     2. a list of strings representing allele names
///     3. a list of strings representing the name of tissues from which the peptide was observed.
/// NOTE: all inputs must have the same length, otherwise Rust code will panic. 
/// 
/// protoem (Dict[str,str]): A dict representing the input to the function which is composite of protein name (uniprot-id) and its sequence.  
/// 
/// path2cashed_database (str): The path to load the cashed database which is a precomputed and cashed data structure used for annotating input proteins. 
///     See online documentation for more details.  
/// 
/// pseudo_sequence (str): The path to load the pseudo-sequences which is table containing the allele name and its pseudo-sequence.
/// 
/// max_len (int): The maximum length of of the peptide, longer sequences are trimmed and shorter sequences are zero-padded. 
/// 
/// threshold (float): A threshold used for generating the set of negative protein for each tissue, for more information about the meaning of this parameter check the
/// online documentation, specifically, the rust function (create_negative_database_from_positive which is defined in the utils module)
/// 
///  fold_neg (int): The amount of negative examples to sample, Xfold the training datasets, where 1 represent a ratio of 1 negative per each positive example while 5 represents  
///  5 negatives from each positive example.
/// 
/// test_size (int): The size of the test dataset, a float in the range (0,1) representing the fraction of the total amount of data 
/// 
/// ### Return
/// A tuple of 4 elements each with the following composition and structure: 
///     Tuple 1: Training tuple which is composite of 4 elements arranges as follow:
///         a. encoded peptide sequence --> a NumPy array of shape (num_mapped_train_peptides,max_len) and a type of np.uint8 representing the encoded peptide sequences   
///         b. encoded pseudo sequence  --> a NumPy array of shape (num_mapped_train_peptides,34) and a type of np.uint8 representing the encoded pseudo sequences   
///         c. encoded gene expression  --> a NumPy array of shape (num_mapped_train_peptides,1) and a type of np.float32 representing the gene expression of the parent transcript.  
///         d. encoded subcellular locations --> a NumPy array of shape (num_mapped_train_peptides,1049) and a type of np.uint8 representing the encoded subcellular location
///         e. encoded labels --> a NumPy of shape (num_mapped_peptides,1) and a type of np.uint8 representing the label of the peptide, where 1 represent binders, 0 represents non-binders
///     
///     Tuple 2: Testing tuples which is composite of 4 elements arranges as follow:
///         a. encoded peptide sequence --> a NumPy array of shape (num_mapped_test_peptides,max_len) and a type of np.uint8 representing the encoded peptide sequences   
///         b. encoded pseudo sequence  --> a NumPy array of shape (num_mapped_test_peptides,34) and a type of np.uint8 representing the encoded pseudo sequences   
///         c. encoded gene expression  --> a NumPy array of shape (num_mapped_test_peptides,1049) and a type of np.float32 representing the gene expression of the parent transcript.
///         d. encoded subcellular locations --> a NumPy array of shape (num_mapped_test_peptides,1049) and a type of np.uint8 representing the encoded subcellular location  
///         e. encoded labels --> a NumPy of shape (num_mapped_test_peptides,1) and a type of np.uint8 representing the label of the peptide, where 1 represent binders, 0 represents non-binders
///     
///     Tuple 3: unmapped train data which is a tuple of four elements:
///         a. A Vector of strings representing the input peptide sequence. 
///         b. A Vector of strings representing the allele names. 
///         c. A Vector of strings representing the tissue names. 
///         d. A Vector of strings representing the peptide labels. 
/// 
///     Tuple 4: unmapped test data which is a tuple of four elements:
///         a. A Vector of strings representing the input peptide sequence. 
///         b. A Vector of strings representing the allele names. 
///         c. A Vector of strings representing the tissue names. 
///         d. A Vector of strings representing the peptide labels. 
#[pyfunction]
pub fn generate_train_based_on_seq_exp_subcell<'py>(py:Python<'py>,
input2prepare:(Vec<String>,Vec<String>,Vec<String>),
            proteome:HashMap<String,String>, path2cashed_db:String, 
            path2pseudo_seq:String, max_len:usize,
            threshold:f32, fold_neg:u32, test_size:f32
    )->(
        (&'py PyArray<u8,Dim<[usize;2]>> /*Train pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train encoded sequence */,
         &'py PyArray<f32,Dim<[usize;2]>> /*Train expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train sub-cellular location values*/,
         &'py PyArray<u8,Dim<[usize;2]>> /* Train labels */),
        
        (&'py PyArray<u8,Dim<[usize;2]>> /*Test pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Test encoded sequence */,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train sub-cellular location values*/,
        &'py PyArray<u8,Dim<[usize;2]>> /* Test labels */),

        (Vec<String>,Vec<String>,Vec<String>, Vec<u8>),// unmapped train data points 
        (Vec<String>,Vec<String>,Vec<String>, Vec<u8>) // unmapped test data points
    )
{
    // Prepare the prelude for the encoders
    //------------------------------------- 
    let (train_data, test_data, pseudo_seq
        ,anno_table)=preparation_prelude(&input2prepare,&proteome,path2cashed_db,path2pseudo_seq,
        threshold,fold_neg,test_size);
    
    // encode and prepare the training data
    //-------------------------------------
    let (encoded_train_data,unmapped_train_data)=prepare_data_for_seq_exp_and_subcellular_data(train_data,
        &pseudo_seq,max_len,&proteome,&anno_table); 
    
    let(encoded_test_data,unmapped_test_data)=prepare_data_for_seq_exp_and_subcellular_data(test_data,
        &pseudo_seq,max_len,&proteome,&anno_table); 

    //return the results
    //------------------
    (
        (encoded_train_data.0.to_pyarray(py),encoded_train_data.1.to_pyarray(py),encoded_train_data.2.to_pyarray(py),encoded_train_data.3.to_pyarray(py),encoded_train_data.4.to_pyarray(py)),
        (encoded_test_data.0.to_pyarray(py),encoded_test_data.1.to_pyarray(py),encoded_test_data.2.to_pyarray(py),encoded_test_data.3.to_pyarray(py),encoded_test_data.4.to_pyarray(py)),
        unmapped_train_data,
        unmapped_test_data
    )
}

/// ### Signature
/// generate_train_based_on_seq_exp(input2prepare:Tuple[List[str], List[str], List[str]], 
///             protoem:Dict[str,str], path2cashed_database:str,
///             pseudo_sequence:str, max_len:int, threshold:float, 
///             fold_neg:int, test_size:float)->Tuple[
///             Tuple[np.ndarray[shape=(Num_mapped_peptides,max_len),type=np.uint8],np.ndarray[shape=(Num_mapped_peptides,34),type=np.uint8]
///                   np.ndarray[shape=(Num_mapped_peptides,1),type=np.float32], np.ndarray[shape=(Num_mapped_peptides,1),type=np.uint8]
///                 ],
///             Tuple[Tuple[np.ndarray[shape=(Num_mapped_peptides,max_len),type=np.uint8],np.ndarray[shape=(Num_mapped_peptides,34),type=np.uint8]
///                   np.ndarray[shape=(Num_mapped_peptides,1),type=np.float32], np.ndarray[shape=(Num_mapped_peptides,1),type=np.uint8]
///                 ],
///             Tuple[List[str],List[str],List[str],List[str]],
///             Tuple[List[str],List[str],List[str],List[str]] 
///     ]
/// 
/// ### Summary
/// A python binder to the omics_builder module written in Rust, enabling efficient and parallel processing and preparation of the input data. 
/// 
/// ### Parameters  
/// input2prepare (Tuple[List[str], List[str], List[str]]): A tuple representing the main inputs to the code, this tuple is composite of three elements 
///     1. a list of strings representing input peptides 
///     2. a list of strings representing allele names
///     3. a list of strings representing the name of tissues from which the peptide was observed.
/// NOTE: all inputs must have the same length, otherwise Rust code will panic. 
/// 
/// protoem (Dict[str,str]): A dict representing the input to the function which is composite of protein name (uniprot-id) and its sequence.  
/// 
/// path2cashed_database (str): The path to load the cashed database which is a precomputed and cashed data structure used for annotating input proteins. 
///     See online documentation for more details.  
/// 
/// pseudo_sequence (str): The path to load the pseudo-sequences which is table containing the allele name and its pseudo-sequence.
/// 
/// max_len (int): The maximum length of of the peptide, longer sequences are trimmed and shorter sequences are zero-padded. 
/// 
/// threshold (float): A threshold used for generating the set of negative protein for each tissue, for more information about the meaning of this parameter check the
/// online documentation, specifically, the rust function (create_negative_database_from_positive which is defined in the utils module)
/// 
///  fold_neg (int): The amount of negative examples to sample, Xfold the training datasets, where 1 represent a ratio of 1 negative per each positive example while 5 represents  
///  5 negatives from each positive example.
/// 
/// test_size (int): The size of the test dataset, a float in the range (0,1) representing the fraction of the total amount of data 
/// 
/// ### Return
/// A tuple of 4 elements each with the following composition and structure: 
///     Tuple 1: Training tuple which is composite of 4 elements arranges as follow:
///         a. encoded peptide sequence --> a NumPy array of shape (num_mapped_train_peptides,max_len) and a type of np.uint8 representing the encoded peptide sequences   
///         b. encoded pseudo sequence  --> a NumPy array of shape (num_mapped_train_peptides,34) and a type of np.uint8 representing the encoded pseudo sequences   
///         c. encoded gene expression  --> a NumPy array of shape (num_mapped_train_peptides,1) and a type of np.float32 representing the gene expression of the parent transcript.  
///         d. encoded subcellular locations --> a NumPy array of shape (num_mapped_train_peptides,1049) and a type of np.uint8 representing the encoded subcellular location
///         e. encoded context vectors --> a NumPy array of shape (num_mapped_train_peptides, Num_genes) and a type of np.float32 representing the encoded context vectors
///         f. encoded labels --> a NumPy of shape (num_mapped_peptides,1) and a type of np.uint8 representing the label of the peptide, where 1 represent binders, 0 represents non-binders
///     
///     Tuple 2: Testing tuples which is composite of 4 elements arranges as follow:
///         a. encoded peptide sequence --> a NumPy array of shape (num_mapped_test_peptides,max_len) and a type of np.uint8 representing the encoded peptide sequences   
///         b. encoded pseudo sequence  --> a NumPy array of shape (num_mapped_test_peptides,34) and a type of np.uint8 representing the encoded pseudo sequences   
///         c. encoded gene expression  --> a NumPy array of shape (num_mapped_test_peptides,1049) and a type of np.float32 representing the gene expression of the parent transcript.
///         d. encoded subcellular locations --> a NumPy array of shape (num_mapped_test_peptides,1049) and a type of np.uint8 representing the encoded subcellular location  
///         e. encoded context vectors --> a NumPy array of shape (num_mapped_test_peptides, Num_genes) and a type of np.float32 representing the encoded context vectors
///         f. encoded labels --> a NumPy of shape (num_mapped_test_peptides,1) and a type of np.uint8 representing the label of the peptide, where 1 represent binders, 0 represents non-binders
///     
///     Tuple 3: unmapped train data which is a tuple of four elements:
///         a. A Vector of strings representing the input peptide sequence. 
///         b. A Vector of strings representing the allele names. 
///         c. A Vector of strings representing the tissue names. 
///         d. A Vector of strings representing the peptide labels. 
/// 
///     Tuple 4: unmapped test data which is a tuple of four elements:
///         a. A Vector of strings representing the input peptide sequence. 
///         b. A Vector of strings representing the allele names. 
///         c. A Vector of strings representing the tissue names. 
///         d. A Vector of strings representing the peptide labels.
#[pyfunction] 
pub fn generate_train_based_on_seq_exp_subcell_context<'py>(py:Python<'py>,
        input2prepare:(Vec<String>,Vec<String>,Vec<String>),
        proteome:HashMap<String,String>, path2cashed_db:String, 
        path2pseudo_seq:String, max_len:usize,threshold:f32,
        fold_neg:u32, test_size:f32
    )->(
        (&'py PyArray<u8,Dim<[usize;2]>> /*Train pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train encoded sequence */,
         &'py PyArray<f32,Dim<[usize;2]>> /*Train expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train sub-cellular location values*/,
         &'py PyArray<f32,Dim<[usize;2]>> /*Train context vectors*/,&'py PyArray<u8,Dim<[usize;2]>> /* Train labels */),
        
        (&'py PyArray<u8,Dim<[usize;2]>> /*Test pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Test encoded sequence */,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train sub-cellular location values*/,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test context vectors*/, &'py PyArray<u8,Dim<[usize;2]>> /* Test labels */),

        (Vec<String>,Vec<String>,Vec<String>, Vec<u8>),// unmapped train data points 
        (Vec<String>,Vec<String>,Vec<String>, Vec<u8>) // unmapped test data points
    )
{
    // Prepare the prelude for the encoders
    //------------------------------------- 
    let (train_data, test_data, pseudo_seq
        ,anno_table)=preparation_prelude(&input2prepare,&proteome,path2cashed_db,path2pseudo_seq,
        threshold,fold_neg,test_size);
    // create the annotations
    //-----------------------
    // encode and prepare the training data
    //-------------------------------------
    let (encoded_train_data,unmapped_train_data)=prepare_data_for_seq_exp_subcellular_and_context_data(train_data,
        &pseudo_seq,max_len,&proteome,&anno_table); 
    
    let(encoded_test_data,unmapped_test_data)=prepare_data_for_seq_exp_subcellular_and_context_data(test_data,
        &pseudo_seq,max_len,&proteome,&anno_table); 

    //return the results
    //------------------
    (
        (encoded_train_data.0.to_pyarray(py),encoded_train_data.1.to_pyarray(py),encoded_train_data.2.to_pyarray(py),encoded_train_data.3.to_pyarray(py),encoded_train_data.4.to_pyarray(py),encoded_train_data.5.to_pyarray(py)),
        (encoded_test_data.0.to_pyarray(py),encoded_test_data.1.to_pyarray(py),encoded_test_data.2.to_pyarray(py),encoded_test_data.3.to_pyarray(py),encoded_test_data.4.to_pyarray(py),encoded_test_data.5.to_pyarray(py)),
        unmapped_train_data,
        unmapped_test_data
    )

}

/// ### Signature
/// generate_train_based_on_seq_exp(input2prepare:Tuple[List[str], List[str], List[str]], 
///             protoem:Dict[str,str], path2cashed_database:str,
///             pseudo_sequence:str, max_len:int, threshold:float, 
///             fold_neg:int, test_size:float)->Tuple[
///             Tuple[np.ndarray[shape=(Num_mapped_peptides,max_len),type=np.uint8],np.ndarray[shape=(Num_mapped_peptides,34),type=np.uint8]
///                   np.ndarray[shape=(Num_mapped_peptides,1),type=np.float32], np.ndarray[shape=(Num_mapped_peptides,1),type=np.uint8]
///                 ],
///             Tuple[Tuple[np.ndarray[shape=(Num_mapped_peptides,max_len),type=np.uint8],np.ndarray[shape=(Num_mapped_peptides,34),type=np.uint8]
///                   np.ndarray[shape=(Num_mapped_peptides,1),type=np.float32], np.ndarray[shape=(Num_mapped_peptides,1),type=np.uint8]
///                 ],
///             Tuple[List[str],List[str],List[str],List[str]],
///             Tuple[List[str],List[str],List[str],List[str]] 
///     ]
/// 
/// ### Summary
/// A python binder to the omics_builder module written in Rust, enabling efficient and parallel processing and preparation of the input data. 
/// 
/// ### Parameters  
/// input2prepare (Tuple[List[str], List[str], List[str]]): A tuple representing the main inputs to the code, this tuple is composite of three elements 
///     1. a list of strings representing input peptides 
///     2. a list of strings representing allele names
///     3. a list of strings representing the name of tissues from which the peptide was observed.
/// NOTE: all inputs must have the same length, otherwise Rust code will panic. 
/// 
/// protoem (Dict[str,str]): A dict representing the input to the function which is composite of protein name (uniprot-id) and its sequence.  
/// 
/// path2cashed_database (str): The path to load the cashed database which is a precomputed and cashed data structure used for annotating input proteins. 
///     See online documentation for more details.  
/// 
/// pseudo_sequence (str): The path to load the pseudo-sequences which is table containing the allele name and its pseudo-sequence.
/// 
/// max_len (int): The maximum length of of the peptide, longer sequences are trimmed and shorter sequences are zero-padded. 
/// 
/// threshold (float): A threshold used for generating the set of negative protein for each tissue, for more information about the meaning of this parameter check the
/// online documentation, specifically, the rust function (create_negative_database_from_positive which is defined in the utils module)
/// 
///  fold_neg (int): The amount of negative examples to sample, Xfold the training datasets, where 1 represent a ratio of 1 negative per each positive example while 5 represents  
///  5 negatives from each positive example.
/// 
/// test_size (int): The size of the test dataset, a float in the range (0,1) representing the fraction of the total amount of data 
/// 
/// ### Return
/// A tuple of 4 elements each with the following composition and structure: 
///     Tuple 1: Training tuple which is composite of 4 elements arranges as follow:
///         a. encoded peptide sequence --> a NumPy array of shape (num_mapped_train_peptides,max_len) and a type of np.uint8 representing the encoded peptide sequences   
///         b. encoded pseudo sequence  --> a NumPy array of shape (num_mapped_train_peptides,34) and a type of np.uint8 representing the encoded pseudo sequences   
///         c. encoded gene expression  --> a NumPy array of shape (num_mapped_train_peptides,1) and a type of np.float32 representing the gene expression of the parent transcript.  
///         d. encoded subcellular locations --> a NumPy array of shape (num_mapped_train_peptides,1049) and a type of np.uint8 representing the encoded subcellular location
///         e. encoded context vectors --> a NumPy array of shape (num_mapped_train_peptides, Num_genes) and a type of np.float32 representing the encoded context vectors
///         f. encoded distance to glycosylation --> a NumPy array of shape (num_mapped_train_peptides, 1) and a type of np.uint32 representing the encoded distance to glycosylation
///         g. encoded labels --> a NumPy of shape (num_mapped_peptides,1) and a type of np.uint8 representing the label of the peptide, where 1 represent binders, 0 represents non-binders
///         
///     
///     Tuple 2: Testing tuples which is composite of 4 elements arranges as follow:
///         a. encoded peptide sequence --> a NumPy array of shape (num_mapped_test_peptides,max_len) and a type of np.uint8 representing the encoded peptide sequences   
///         b. encoded pseudo sequence  --> a NumPy array of shape (num_mapped_test_peptides,34) and a type of np.uint8 representing the encoded pseudo sequences   
///         c. encoded gene expression  --> a NumPy array of shape (num_mapped_test_peptides,1049) and a type of np.float32 representing the gene expression of the parent transcript.
///         d. encoded subcellular locations --> a NumPy array of shape (num_mapped_test_peptides,1049) and a type of np.uint8 representing the encoded subcellular location  
///         e. encoded context vectors --> a NumPy array of shape (num_mapped_test_peptides, Num_genes) and a type of np.float32 representing the encoded context vectors
///         f. encoded distance to glycosylation --> a NumPy array of shape (num_mapped_test_peptides, 1) and a type of np.uint32 representing the encoded distance to glycosylation
///         g. encoded labels --> a NumPy of shape (num_mapped_test_peptides,1) and a type of np.uint8 representing the label of the peptide, where 1 represent binders, 0 represents non-binders
///     
///     Tuple 3: unmapped train data which is a tuple of four elements:
///         a. A Vector of strings representing the input peptide sequence. 
///         b. A Vector of strings representing the allele names. 
///         c. A Vector of strings representing the tissue names. 
///         d. A Vector of strings representing the peptide labels. 
/// 
///     Tuple 4: unmapped test data which is a tuple of four elements:
///         a. A Vector of strings representing the input peptide sequence. 
///         b. A Vector of strings representing the allele names. 
///         c. A Vector of strings representing the tissue names. 
///         d. A Vector of strings representing the peptide labels. 
#[pyfunction]
pub fn generate_train_based_on_seq_exp_subcell_loc_context_d2g<'py>(py:Python<'py>,
            input2prepare:(Vec<String>,Vec<String>,Vec<String>),
            proteome:HashMap<String,String>, path2cashed_db:String, 
            path2pseudo_seq:String, max_len:usize,
            threshold:f32, fold_neg:u32, test_size:f32
    )->(
        (&'py PyArray<u8,Dim<[usize;2]>> /*Train peptide sequences*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train pseudo-sequences*/,
         &'py PyArray<f32,Dim<[usize;2]>> /*Train expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train sub-cellular location values*/,
         &'py PyArray<f32,Dim<[usize;2]>> /*Train context vectors*/, &'py PyArray<u32,Dim<[usize;2]>> /*Train distance to glycosylation*/,
         &'py PyArray<u8,Dim<[usize;2]>> /* Train labels */),
        
        (&'py PyArray<u8,Dim<[usize;2]>> /*Test pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Test encoded sequence */,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Test sub-cellular location values*/,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test context vectors*/,&'py PyArray<u32,Dim<[usize;2]>> /*Test distance to glycosylation*/,
        &'py PyArray<u8,Dim<[usize;2]>> /* Test labels */), 

        (Vec<String>,Vec<String>,Vec<String>, Vec<u8>),// unmapped train data points 
        (Vec<String>,Vec<String>,Vec<String>, Vec<u8>) // unmapped test data points
    )
{
    // Prepare the prelude for the encoders
    //------------------------------------- 
    let (train_data, test_data, pseudo_seq
        ,anno_table)=preparation_prelude(&input2prepare,&proteome,path2cashed_db,path2pseudo_seq,
        threshold,fold_neg,test_size);
    
    // encode and prepare the training data
    //-------------------------------------
    let (encoded_train_data,unmapped_train_data)=prepare_data_for_seq_exp_subcellular_context_d2g_data(train_data,
        &pseudo_seq,max_len,&proteome,&anno_table); 
    
    let(encoded_test_data,unmapped_test_data)=prepare_data_for_seq_exp_subcellular_context_d2g_data(test_data,
        &pseudo_seq,max_len,&proteome,&anno_table); 
    
    //return the results
    //------------------
    (
        (encoded_train_data.0.to_pyarray(py),encoded_train_data.1.to_pyarray(py),encoded_train_data.2.to_pyarray(py),
        encoded_train_data.3.to_pyarray(py),encoded_train_data.4.to_pyarray(py),encoded_train_data.5.to_pyarray(py),encoded_train_data.6.to_pyarray(py)),
        (encoded_test_data.0.to_pyarray(py),encoded_test_data.1.to_pyarray(py),encoded_test_data.2.to_pyarray(py),
        encoded_test_data.3.to_pyarray(py),encoded_test_data.4.to_pyarray(py),encoded_test_data.5.to_pyarray(py),encoded_test_data.6.to_pyarray(py)),
        unmapped_train_data,
        unmapped_test_data
    )
}