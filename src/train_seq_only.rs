// A collection of function for training and preparing the data for sequence only models 
//---------------------------------------------------------------------------------------
// Load the library modules 
//-------------------------
use pyo3::prelude::*;
use ndarray::{Dim, Array};
use numpy::{PyArray, ToPyArray};
use rand::prelude::IteratorRandom;
use crate::peptides::{generate_a_train_db_by_shuffling_rs,encode_sequence_rs};
use crate::sequence_builder::{prepare_train_ds_proteome_sampling, prepare_train_ds_same_protein_sampling, prepare_train_ds_shuffling, prepare_train_ds_expressed_protein_sampling}; 
use rand;
use std::path::Path;
use std::collections::HashMap;
use crate::utils::*; 


/// ### Signature
/// prepare_train_ds_shuffling_sm(positive_examples:List[str],fold_neg:int,max_len:int,test_size:float)->Tuple[
///                                                                                     Tuple[np.ndarray,np.ndarray],
///                                                                                     Tuple[np.ndarray,np.ndarray]]
/// ### Summary
/// A rust-optimized function that can be used for sequence only models the function generates negative examples using shuffling 
/// 
/// ### Parameters
/// positive_examples: List of string representing positive peptides 
/// fold_neg: The fold of negative examples an integer representing the ration of positives to negative, if 1 then 1 negative is generated for every positive
///     meanwhile if it is set to 10 then 10 negative examples are generated from every positive examples.  
/// max_len: The maximum length to transfer a variable length peptides into a fixed length peptides. Here, shorter peptides are zero padded into this predefined 
/// sequence length. while longer peptides are trimmed into this length. 
/// test_size: The size of the test dataset, a float in the range (0,1)
/// 
/// ## Returns 
/// A tuple of two tuples with the following structure
///                         train_tuple:
///                               |---> encoded_train_seq:    a tensor with shape(num_train_examples,max_length) and type u8
///                               |---> encoded_train_label:  a tensor with shape(num_train_examples,1) and type u8
///                         test_tuple:
///                               |---> encoded_test_seq:     a tensor with shape(num_train_examples,max_length)and type u8
///                               |---> encoded_test_label:  a tensor with shape(num_train_examples,1) and type u8
/// -----------------------------------------------
#[pyfunction]
pub fn generate_train_ds_shuffling_sm<'py>(py:Python<'py>, 
            positive_examples:Vec<String>,fold_neg:u32,max_len:usize,test_size:f32)->(
                (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>>),
                (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>>)
            )
{
    // get the encoded data
    //---------------------
    let(train_database,test_database)=prepare_train_ds_shuffling(&positive_examples,fold_neg,test_size); 

    // numerically encode the datasets
    //--------------------------------
    let encoded_test_seq=encode_sequence_rs(test_database.0,max_len).to_pyarray(py);                           
    let encoded_test_labels= Array::from_shape_vec((encoded_test_seq.shape()[0],1), test_database.1).unwrap().to_pyarray(py);


    let encoded_train_seq=encode_sequence_rs(train_database.0,max_len).to_pyarray(py);                           
    let encoded_train_labels= Array::from_shape_vec((encoded_train_seq.shape()[0],1), train_database.1).unwrap().to_pyarray(py);

    (
        (encoded_train_seq,encoded_train_labels),
        (encoded_test_seq,encoded_test_labels)
    )
}

/// ### Signature
/// generate_a_train_arrays_by_shuffling(positive_examples:List[str], fold_neg:int, max_len:int)->Tuple[Lnp.ndarray,np.ndarray]
/// ### Summary
/// Takes a list of positive examples, samples an x fold of negative examples through shuffling where x is determined through the parameters fold_neg
/// and returns two arrays the first represent the encoded representation for input peptides while the second represent the label of each peptide in the database. 
/// ### Executioner 
/// This function is a wrapper function for two main Rust functions: generate_a_train_db_by_shuffling_rs and encode_sequence_rs, for more details regard the execution logic check 
/// the documentation for these functions.
/// ### Parameters
/// positive_examples: A list of peptide sequences representing positive examples
/// fold_neg: The ration of the negative, i.e. shuffled generated, to the positive examples, e.g. 1 means one positive to 1 negative while 10 means 1 positive to 10 negatives.
/// max_len: The maximum peptide length, used for padding, longer peptides are trimmed to this length and shorter are zero-padded.   
/// ### Results
/// A tuple of two arrays, the first represents encoded peptide sequences which has a shape of (num_peptides, max_length) as a type of u8 while the second array
/// represent the numerical labels of each peptides, 1 represent positive peptides and 0 represent negative peptides. This array has a shape of (num_peptides,1) and u8 as types. 
/// ### Usage
/// For training a sequence only model on the whole dataset using shuffled negative when a test dataset is not required. e.g in fine-tunning or final training, 
/// -----------------------------------------------
#[pyfunction]
fn generate_a_train_arrays_by_shuffling<'py>(py:Python<'py>,
        positive_examples:Vec<String>,
        fold_neg:u32, max_len:usize)->(&'py PyArray<u8,Dim<[usize;2]>>,&'py PyArray<u8,Dim<[usize;2]>>)
{
    // create the database which is made from the sequences and the labels
    let (generated_seq, labels) = generate_a_train_db_by_shuffling_rs(positive_examples,fold_neg);

    // numerical encode the database
    let encoded_seq = encode_sequence_rs(generated_seq,max_len).to_pyarray(py);
    
    // create an array from the labels
    let labels = Array::from_shape_vec((encoded_seq.shape()[0],1), labels).unwrap().to_pyarray(py);

    // return the results
    (encoded_seq,labels)
}


/// ### Signature
/// generate_train_ds_proteome_sampling_sm(positive_examples:List[str],proteome:Dict[str,str],
///                                         fold_neg:int,max_len:int,test_size:float)->Tuple[
///                                                                                     Tuple[np.ndarray,np.ndarray],
///                                                                                     Tuple[np.ndarray,np.ndarray]]
/// ### Summary
/// A rust-optimized function that can be used for sequence only models the function generates negative examples using random sampling from the proteome 
/// 
/// ### Parameters
/// positive_examples: List of string representing positive peptides 
/// proteome: a dict of protein name and protein sequences
/// fold_neg: The fold of negative examples an integer representing the ration of positives to negative, if 1 then 1 negative is generated for every positive
///     meanwhile if it is set to 10 then 10 negative examples are generated from every positive examples.  
/// max_len: The maximum length to transfer a variable length peptides into a fixed length peptides. Here, shorter peptides are zero padded into this predefined 
/// sequence length. while longer peptides are trimmed into this length. 
/// test_size: The size of the test dataset, a float in the range (0,1)
/// 
/// ## Returns 
/// A tuple of two tuples with the following structure
///                         train_tuple:
///                               |---> encoded_train_seq:    a tensor with shape(num_train_examples,max_length) and type u8
///                               |---> encoded_train_label:  a tensor with shape(num_train_examples,1) and type u8
///                         test_tuple:
///                               |---> encoded_test_seq:     a tensor with shape(num_train_examples,max_length)and type u8
///                               |---> encoded_test_label:  a tensor with shape(num_train_examples,1) and type u8
/// -----------------------------------------------
#[pyfunction]
pub fn generate_train_ds_proteome_sampling_sm<'py>(py:Python<'py>, 
            positive_examples:Vec<String>,proteome:HashMap<String,String>,fold_neg:u32,max_len:usize,test_size:f32)->(
                (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>>),
                (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>>)
            )
{
    let(train_database,test_database)=prepare_train_ds_proteome_sampling(&positive_examples,&proteome,fold_neg,test_size);
    // numerically encode the datasets
    //--------------------------------
    let encoded_test_seq=encode_sequence_rs(test_database.0,max_len).to_pyarray(py);                           
    let encoded_test_labels= Array::from_shape_vec((encoded_test_seq.shape()[0],1), test_database.1).unwrap().to_pyarray(py);


    let encoded_train_seq=encode_sequence_rs(train_database.0,max_len).to_pyarray(py);                           
    let encoded_train_labels= Array::from_shape_vec((encoded_train_seq.shape()[0],1), train_database.1).unwrap().to_pyarray(py);
    (
        (encoded_train_seq,encoded_train_labels),
        (encoded_test_seq,encoded_test_labels)
    )
}

/// ### Signature
/// generate_train_ds_same_protein_sampling_sm(positive_examples:List[str],proteome:Dict[str,str],
///                                         fold_neg:int,max_len:int,test_size:float)->Tuple[
///                                                                                     Tuple[np.ndarray,np.ndarray],
///                                                                                     Tuple[np.ndarray,np.ndarray]]
/// ### Summary
/// A rust-optimized function that can be used for sequence only models the function generates negative examples through sampling from the same protein 
/// 
/// ### Parameters
/// positive_examples: List of string representing positive peptides 
/// proteome: a dict of protein name and protein sequences
/// fold_neg: The fold of negative examples an integer representing the ration of positives to negative, if 1 then 1 negative is generated for every positive
///     meanwhile if it is set to 10 then 10 negative examples are generated from every positive examples.  
/// max_len: The maximum length to transfer a variable length peptides into a fixed length peptides. Here, shorter peptides are zero padded into this predefined 
/// sequence length. while longer peptides are trimmed into this length. 
/// test_size: The size of the test dataset, a float in the range (0,1)
/// 
/// ## Returns 
/// A tuple of two tuples with the following structure
///                         train_tuple:
///                               |---> encoded_train_seq:    a tensor with shape(num_train_examples,max_length) and type u8
///                               |---> encoded_train_label:  a tensor with shape(num_train_examples,1) and type u8
///                         test_tuple:
///                               |---> encoded_test_seq:     a tensor with shape(num_train_examples,max_length)and type u8
///                               |---> encoded_test_label:  a tensor with shape(num_train_examples,1) and type u8
/// -----------------------------------------------
#[pyfunction]
pub fn generate_train_ds_same_protein_sampling_sm<'py>(py:Python<'py>, 
            positive_examples:Vec<String>,proteome:HashMap<String,String>,fold_neg:u32,
            max_len:usize,test_size:f32)->(
                (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>>),
                (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>>)
            )
{
    // prepare the train and test datasets through sampling from the same protein 
    let(train_database,test_database)=prepare_train_ds_same_protein_sampling(&positive_examples,&proteome,fold_neg,test_size);
    // numerically encode the datasets
    //--------------------------------
    let encoded_test_seq=encode_sequence_rs(test_database.0,max_len).to_pyarray(py);                           
    let encoded_test_labels= Array::from_shape_vec((encoded_test_seq.shape()[0],1), test_database.1).unwrap().to_pyarray(py);


    let encoded_train_seq=encode_sequence_rs(train_database.0,max_len).to_pyarray(py);                           
    let encoded_train_labels= Array::from_shape_vec((encoded_train_seq.shape()[0],1), train_database.1).unwrap().to_pyarray(py);

    (
        (encoded_train_seq,encoded_train_labels),
        (encoded_test_seq,encoded_test_labels)
    )
}

/// ### Signature
/// generate_train_ds_expressed_protein_sampling_sm(positive_examples:List[str],proteome:Dict[str,str],path2expression_map:str,
///                     tissue_name:str,threshold:float, 
///                     fold_neg:int, max_len:int, test_size:float)->Tuple[
///                                                                                     Tuple[np.ndarray,np.ndarray],
///                                                                                     Tuple[np.ndarray,np.ndarray]]
/// ### Summary
/// A rust-optimized function that can be used for sequence only models, the function generates negative examples through sampling from the same protein 
/// 
/// ### Parameters
/// positive_examples: List of string representing positive peptides 
/// proteome: a dict of protein name and protein sequences
/// path2expression_map: The path to load gene expression datasets
/// tissue_name: The name of the tissue to be restrict the gene expression to 
/// threshold: The minimum expression level of each protein
/// fold_neg: The fold of negative examples an integer representing the ration of positives to negative, if 1 then 1 negative is generated for every positive
///     meanwhile if it is set to 10 then 10 negative examples are generated from every positive examples.  
/// max_len: The maximum length to transfer a variable length peptides into a fixed length peptides. Here, shorter peptides are zero padded into this predefined 
/// sequence length. while longer peptides are trimmed into this length. 
/// test_size: The size of the test dataset, a float in the range (0,1)
/// 
/// ## Returns 
/// A tuple of two tuples with the following structure
///                         train_tuple:
///                               |---> encoded_train_seq:    a tensor with shape(num_train_examples,max_length) and type u8
///                               |---> encoded_train_label:  a tensor with shape(num_train_examples,1) and type u8
///                         test_tuple:
///                               |---> encoded_test_seq:     a tensor with shape(num_train_examples,max_length)and type u8
///                               |---> encoded_test_label:  a tensor with shape(num_train_examples,1) and type u8
/// ----------------------------------------------- 
#[pyfunction]
pub fn generate_train_ds_expressed_protein_sampling_sm<'py>(py:Python<'py>, 
            positive_examples:Vec<String>,proteome:HashMap<String,String>,
            path2expression_map:String, tissue_name:String, threshold:f32,
            fold_neg:u32,max_len:usize,test_size:f32)->(
                (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>>),
                (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>>)
            )
{
    // prepare the train and test datasets through sampling from the same protein 
    let(train_database,test_database)=prepare_train_ds_expressed_protein_sampling(&positive_examples,&proteome,
        Path::new(&path2expression_map),&tissue_name,threshold,
        fold_neg,test_size);

    // numerically encode the datasets
    //--------------------------------
    let encoded_test_seq=encode_sequence_rs(test_database.0,max_len).to_pyarray(py);                           
    let encoded_test_labels= Array::from_shape_vec((encoded_test_seq.shape()[0],1), test_database.1).unwrap().to_pyarray(py);


    let encoded_train_seq=encode_sequence_rs(train_database.0,max_len).to_pyarray(py);                           
    let encoded_train_labels= Array::from_shape_vec((encoded_train_seq.shape()[0],1), train_database.1).unwrap().to_pyarray(py);

    (
        (encoded_test_seq,encoded_test_labels),
        (encoded_train_seq,encoded_train_labels)
    )
}
/// ### Signature and parameters 
/// generate_train_ds_pm(positive_examples:Dict[str,List[str]], # a dict containing the allele names as keys and list of bound peptides as values
///     proteome:Dict[str,str]|None, # A dict containing protein id as keys and protein sequences as values, can be none is shuffle is used
///     path2gene_expression: str| None, # A string representing the path to the expression table, must not be none, if sampling method is exp_samp
///     tissue_name: str|None,           # The name of the tissue, must not be none, if sampling method is exp_samp 
///     path2pseudoSeq: str,             # The path2pseudoSeq, the path to load the pseudo sequences
///     fold_neg:int,                    # The fold of negative examples an integer representing the ration of positives to negative, if 1 then 1 negative is generated for every positive meanwhile if it is set to 10 then 10 negative examples are generated from every positive examples.
///     exp_threshold:float|None,        # The minimum expression level of each protein
///     max_len:int                      # The maximum length to transfer a variable length peptides into a fixed length peptides. Here, shorter peptides are zero padded into this predefined 
///     test_size:float,                 # The size of the test dataset, a float in the range (0,1)
///     method:str                       # The generation method, can be any off: {shuffle,prot_sample,same,exp_samp}
///     )->Tuple[
///                     [np.ndarray,np.ndarray,np.ndarray], # The train arrays tuple, with the first elements being the encoded train array, the encoded pseudo sequences, the encoded numerical labels 
///                     [np.ndarray,np.ndarray,np.ndarray] # The test arrays tuple, with the first elements being the encoded train array, the encoded pseudo sequences, the encoded numerical labels
///                     ]
/// --------------
#[pyfunction]
pub fn generate_train_ds_pm<'py>(py:Python<'py>, 
            positive_examples:HashMap<String,Vec<String>>,
            proteome:Option<HashMap<String,String>>,
            path2gene_expression:Option<String>,
            tissue_name:Option<String>,
            path2pseudo_seq:String,
            fold_neg:u32,
            exp_threshold:Option<f32>,
            max_len:usize,
            test_size:f32,
             method:String)->(
                (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>> ,&'py PyArray<u8,Dim<[usize;2]>>),
                (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>>)
            )
{
    // check that the input is plausible 
    if positive_examples.len()==0{panic!("positive examples are empty!!!");}
    if test_size >=1.0 || test_size<=0.0 {panic!("your test size: {} is out on range, it must be a value between [0,1)",test_size)}

    // create a functional pointers to methods
    //----------------------------------------
    match method.as_str()
    {
        "shuffle"=>(),
        "prot_sample"=>
        {
            match &proteome
            {
                Some(_)=>(),
                None=>panic!("Method: prot_sample can not be used when the input proteome has not been provided")
            }
        }, 
        "same"=>
        {
            match &proteome
            {
                Some(_)=>(),
                None=>panic!("Method: same can not be used when the input proteome has not been provided")
            }
        },
        "exp_samp"=>
        {
            match &proteome
            {
                Some(_)=>(),
                None=>panic!("Method: same can not be used when the input proteome has not been provided")
            }
            match &path2gene_expression
            {
                Some(_)=>(),
                None=>panic!("Method: exp_samp can not be used when the input expression table has not been provided")
            }
            match &tissue_name
            {
                Some(_)=>(),
                None=>panic!("Method: exp_samp can not be used when the input tissue name has not been provided")
            }
        }
        _=>panic!("Method: {} is currently not supported, only: random, prot_sample, same or exp_samp are supported",method)
    };
    // Read the pseudo-sequences
    let pseudo_seq=read_pseudo_seq(Path::new(&path2pseudo_seq)); 

    // Count the number of entries in the HashMap to generate the results
    //-------------------------------------------------------------------
    let num_entries=fold_neg*positive_examples
                .iter()
                .map(|(_,peptides)| if peptides.len() > 100{return peptides.len() as u32;} else {return 0;})
                .sum::<u32>(); 
    //-----------------------------------------------------
    let num_test_examples=(test_size*num_entries as f32) as u32;
    let num_train_examples=(num_entries-num_test_examples) as usize;
    
    // allocate vectors to hold the results
    //-------------------------------------
    let (mut train_labels,mut train_alleles,mut train_sequences)=(Vec::with_capacity(num_train_examples),
                    Vec::with_capacity(num_train_examples),Vec::with_capacity(num_train_examples));   

    let (mut test_labels,mut test_alleles,mut test_sequences)=(Vec::with_capacity(num_test_examples as usize),
    Vec::with_capacity(num_test_examples as usize),Vec::with_capacity(num_test_examples as usize));   
    //-----------

    // loop over alleles and generate 
    //-------------------------------
    for (allele_name,peptides) in positive_examples.iter()
    {
        if peptides.len()<100{continue;}// check that the number of peptides is above 50 and skip it otherwise

        let ((mut train_seq,mut train_label),(mut test_seq,mut test_label))=match method.as_str()
        {
            "shuffle"=>prepare_train_ds_shuffling(peptides,fold_neg,test_size),
            "prot_sample"=>prepare_train_ds_proteome_sampling(peptides,proteome.as_ref().unwrap(),fold_neg,test_size), 
            "same"=>prepare_train_ds_same_protein_sampling(peptides,&proteome.as_ref().unwrap(),fold_neg,test_size),
            "exp_samp"=>prepare_train_ds_expressed_protein_sampling(peptides,&proteome.as_ref().unwrap(),
                            Path::new(&path2gene_expression.as_ref().unwrap()),&tissue_name.as_ref().unwrap(),exp_threshold.unwrap(),
                            fold_neg,test_size),
            _=>panic!("Un supported methods")
        };

        // Extend the allocated vectors
        // extend the training arrays
        //---------------------------
        train_alleles.append(&mut vec![pseudo_seq.get(allele_name).unwrap().clone();train_seq.len()]);
        train_sequences.append(&mut train_seq); 
        train_labels.append(&mut train_label);
        // extend the test arrays
        //-----------------------
        test_alleles.append(&mut vec![pseudo_seq.get(allele_name).unwrap().clone();test_seq.len()]);
        test_sequences.append(&mut test_seq); 
        test_labels.append(&mut test_label);
    }
    // Numerically encode the datasets
    //--------------------------------
    let encoded_train_labels= Array::from_shape_vec((train_sequences.len(),1), train_labels).unwrap().to_pyarray(py);
    let encoded_train_peptide_seq=encode_sequence_rs(train_sequences,max_len).to_pyarray(py);
    let encoded_train_pseudo_seq=encode_sequence_rs(train_alleles,34).to_pyarray(py);

    let encoded_test_labels= Array::from_shape_vec((test_sequences.len(),1), test_labels).unwrap().to_pyarray(py);
    let encoded_test_peptide_seq=encode_sequence_rs(test_sequences,max_len).to_pyarray(py);
    let encoded_test_pseudo_seq=encode_sequence_rs(test_alleles,34).to_pyarray(py);  
    
    // return the results 
    //-------------------
    (
        (encoded_train_peptide_seq,encoded_train_pseudo_seq,encoded_train_labels),
        (encoded_test_peptide_seq,encoded_test_pseudo_seq,encoded_test_labels)
    )
}

/// ### Signature
/// generate_train_ds_Qd(path2load_ds:str,path2pseudo_seq:str,
///          test_size:float, max_len: float)-> Tuple[
///                 Tuple[np.ndarray,np.ndarray,np.ndarry],
///                 Tuple[np.ndarray,np.ndarray,np.ndarry],
///            ]
/// ### Parameters
/// path2load_ds: The path to load the database, which is expected to be a table of three column, as follow:
///     1. HLA --> which describe the allele names using the standard notation, e.g. DPA1*01:03-DPB1*02:01
///     2. sequence --> which describe the peptide sequence
///     3. length --> which describe the length of the peptide
///     4. the log IC50 --> which describe the normalized IC50 scores.  
/// 
/// path2pseudo_seq: Which is a description of the path to load the pseudo-sequences. 
/// test_size: The size of the test size, a float between in the range (0,1) i.e. bigger than 0 and smaller than 1, representing the fraction of the test dataset. 
/// max_len: The maximum length to transfer a variable length peptides into a fixed length peptides. Here, shorter peptides are zero padded into this predefined.
/// ### Returns
///     A tuple of tuple representing the training and the testing datasets, each of these tuple contains three arrays:
///     1. train_array: A NumPy array of type np.uint8 and shape of (num_train_examples | num_test_examples, max_length) representing the encoded peptide sequences.  
///     2. pseudo_seq: A NumPy array of type np.uint8 and shape of (num_train_examples | num_test_examples,32) representing the encoded HLA pseudo-sequence.  
///     3. labels: A NumPy array of type np.float32 and shape of (num_train_examples | num_test_examples,f32) representing the test_size.
#[pyfunction]
pub fn generate_train_ds_Qd<'py>(py:Python<'py>,path2load_ds:String, path2pseudo_seq:String, test_size:f32, max_len:usize)->(
    (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>> ,&'py PyArray<f32,Dim<[usize;2]>>),
    (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<f32,Dim<[usize;2]>>)
)
{
    // Check the test size is valid 
    if test_size<=0.0 || test_size>1.0{panic!("In-valid test name: expected the test-size to be a float bigger than 0.0 and smaller than 1.0, however, the test size is: {}",test_size);}
    // Load the training quantitative datasets 
    //-----------------------------------------
    let(allele_names,peptides,affinity)=read_Q_table(Path::new(&path2load_ds));  
    
    // Compute the size of the test dataset 
    //-------------------------------------
    let num_dp=allele_names.len();
    let num_test_dp=(num_dp as f32 *test_size) as usize; 

    // sample the index of the test datasets
    //--------------------------------------
    let mut rng=rand::thread_rng(); 
    let test_peptides_index=(0..allele_names.len()).choose_multiple(&mut rng, num_test_dp); 

    // Extract the index of the binding and non-binding peptides 
    //----------------------------------------------------------
    let (mut test_alleles,mut test_peptides,mut test_affinity)=(Vec::with_capacity(num_test_dp),
                                    Vec::with_capacity(num_test_dp),Vec::with_capacity(num_test_dp));
    // loop over all elements
    //-----------------------
    for elem in test_peptides_index.iter()
    {
        test_alleles.push(allele_names[*elem].to_string()); 
        test_peptides.push(peptides[*elem].to_string());
        test_affinity.push(affinity[*elem]);
    }
    
    // Allocate the training array
    //--------------------------------------------------
    let (mut train_alleles,mut train_peptides,mut train_affinity)=(Vec::with_capacity(num_dp),
                                    Vec::with_capacity(num_dp),Vec::with_capacity(num_dp));

    // Filling the array using the computed data
    //----------------------------------------
    for elem in 0..num_dp
    {
        if !test_peptides_index.contains(&elem)
        {
            train_alleles.push(allele_names[elem].to_string()); 
            train_peptides.push(peptides[elem].to_string());
            train_affinity.push(affinity[elem]);
        }
    }
    // load the pseudo-sequences 
    let pseudo_seq_map=read_pseudo_seq(&Path::new(&path2pseudo_seq));
    // get the sequence of the target alleles
    //---------------------------------------
    let train_alleles= train_alleles.iter().map(|allele|pseudo_seq_map.get(allele).unwrap().clone()).collect::<Vec<_>>();
    let test_alleles= test_alleles.iter().map(|allele|pseudo_seq_map.get(allele).unwrap().clone()).collect::<Vec<_>>();
    // numerically encode the results
    //--------------------------------
   // Numerically encode the datasets
    //--------------------------------
    let encoded_train_labels= Array::from_shape_vec((train_alleles.len(),1), train_affinity).unwrap().to_pyarray(py);
    let encoded_train_peptide_seq=encode_sequence_rs(train_peptides,max_len).to_pyarray(py);
    let encoded_train_pseudo_seq=encode_sequence_rs(train_alleles,34).to_pyarray(py);

    let encoded_test_labels= Array::from_shape_vec((test_peptides.len(),1), test_affinity).unwrap().to_pyarray(py);
    let encoded_test_peptide_seq=encode_sequence_rs(test_peptides,max_len).to_pyarray(py);
    let encoded_test_pseudo_seq=encode_sequence_rs(test_alleles,34).to_pyarray(py);  

    // return the results 
    //-------------------
    (
        (encoded_train_peptide_seq,encoded_train_pseudo_seq,encoded_train_labels),
        (encoded_test_peptide_seq,encoded_test_pseudo_seq,encoded_test_labels)
    )
}

