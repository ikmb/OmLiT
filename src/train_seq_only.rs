// A collection of function for training and preparing the data for sequence only models 
//---------------------------------------------------------------------------------------
// Load the library modules 
//-------------------------
use ndarray::{Dim, Array};
use numpy::{PyArray, ToPyArray};
use pyo3::{pyfunction,Python};
use crate::peptides::{group_by_9mers_rs,generate_a_train_db_by_shuffling_rs,encode_sequence_rs, group_peptides_by_parent_rs,generate_negative_by_sampling_rs}; 
use rand;
use rand::seq::SliceRandom;
use std::{collections::HashMap, hash::Hash};
use crate::{geneExpressionIO::*, expression_db};
use crate::utils::*; 

/// ### Summary
/// The working engine for generating training datasets through shuffling
/// ### Parameters
/// positive_examples: List of string representing positive peptides 
/// fold_neg: The fold of negative examples an integer representing the ration of positives to negative, if 1 then 1 negative is generated for every positive
///     meanwhile if it is set to 10 then 10 negative examples are generated from every positive examples.  
/// test_size: The size of the test dataset, a float in the range (0,1)
/// ## Returns 
/// A tuple of two tuples with the following structure
///                         train_tuple:
///                               |---> train_seq:    A vector of string containing the training peptide sequences
///                               |---> train_label:  A vector of u8 containing or representing train label
///                         test_tuple:
///                               |---> test_seq: A vector of string containing the testing peptide sequences
///                               |---> test_label:   A vector of u8 containing or representing test label
fn prepare_train_ds_shuffling(positive_examples:&Vec<String>,fold_neg:u32,test_size:f32)->((Vec<String>, Vec<u8>),(Vec<String>, Vec<u8>))
{
    
    // check the input is correct
    if positive_examples.len()==0{panic!("Input collection of positive examples is empty");}
    if test_size >=1.0 || test_size<=0.0 {panic!("your test size: {} is out on range, it must be a value between [0,1)",test_size)}

    // create the number of examples
    let num_test_examples=(test_size*positive_examples.len() as f32 ) as usize;
    let num_train_examples= positive_examples.len() as usize - num_test_examples;
    // first let's group by 9 mers
    let unique_9_mers=group_by_9mers_rs(positive_examples)
            .into_iter()
            .collect::<Vec<_>>();
            
    // create a thread for splitting the dataset 
    let mut rng=rand::thread_rng(); 

    // Split the data into test and train
    //------------------------------- 
    let test_9mers=unique_9_mers
                                                    .choose_multiple(&mut rng,num_test_examples)
                                                    .map(|(mer,peptides)|(mer.clone(),peptides.clone()))
                                                    .collect::<Vec<_>>();
    let train_9mers=unique_9_mers
                                                    .iter()
                                                    .filter(|(mer,peptides)| 
                                                            {
                                                                for (test_mer, _) in test_9mers.iter()
                                                                {
                                                                    if mer ==test_mer{return false}
                                                                }
                                                                true
                                                            }
                                                        )
                                                    .map(|(mer,peptides)|(mer.clone(),peptides.clone()))
                                                    .collect::<Vec<_>>();
    // Get train and test peptides
    //----------------------------
    let test_peptides=test_9mers
                        .into_iter()
                        .map(|(mer,peptides)| peptides)
                        .flatten()
                        .collect::<Vec<_>>();
    
    let train_peptides=train_9mers
                        .into_iter()
                        .map(|(mer,peptides)| peptides)
                        .flatten()
                        .collect::<Vec<_>>();
    // Prepare and sample the dataset 
    //-------------------------------
    let test_database=generate_a_train_db_by_shuffling_rs(test_peptides,fold_neg);
    let train_database=generate_a_train_db_by_shuffling_rs(train_peptides,fold_neg); 
    
    // return the results 
    //-------------------
    (train_database,test_database)
}
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
        (encoded_test_seq,encoded_test_labels),
        (encoded_train_seq,encoded_train_labels)
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

/// ### Summary
/// The working engine for generating training datasets through proteome sampling 
/// ### Parameters
/// positive_examples: List of string representing positive peptides 
/// fold_neg: The fold of negative examples an integer representing the ration of positives to negative, if 1 then 1 negative is generated for every positive
///     meanwhile if it is set to 10 then 10 negative examples are generated from every positive examples.   
/// test_size: The size of the test dataset, a float in the range (0,1)
/// ## Returns 
/// A tuple of two tuples with the following structure
///                         train_tuple:
///                               |---> train_seq:    A vector of string containing the training peptide sequences
///                               |---> train_label:  A vector of u8 containing or representing train label
///                         test_tuple:
///                               |---> test_seq: A vector of string containing the testing peptide sequences
///                               |---> test_label:   A vector of u8 containing or representing test label
fn prepare_train_ds_proteome_sampling(positive_examples:&Vec<String>,proteome:&HashMap<String,String>,
        fold_neg:u32,test_size:f32)->((Vec<String>, Vec<u8>),(Vec<String>, Vec<u8>))
{
    
    // check the input is correct
    if positive_examples.len()==0{panic!("Input collection of positive examples is empty");}
    if test_size >=1.0 || test_size<=0.0 {panic!("your test size: {} is out on range, it must be a value between [0,1)",test_size)}

    // create the number of examples
    let num_test_examples=(test_size*positive_examples.len() as f32 ) as usize;
    let num_train_examples= positive_examples.len() as usize - num_test_examples;
    // first let's group by 9 mers
    let unique_9_mers=group_by_9mers_rs(positive_examples)
            .into_iter()
            .collect::<Vec<_>>();
    
    // get parent proteins and remove bound peptides
    //----------------------------------------------
    let positive_parent=group_peptides_by_parent_rs(positive_examples, &proteome)
        .into_iter()        
        .map(|(pep,parents)|parents)
        .flatten()
        .collect::<Vec<_>>(); 

    // filter the database from positive proteins
    //-------------------------------------------
    let target_proteome=proteome
        .into_iter()
        .filter(|(name,seq)| !positive_parent.contains(name))
        .collect::<HashMap<_,_>>(); 


    // create a thread for splitting the dataset 
    let mut rng=rand::thread_rng(); 

    // Split the data into test and train
    //------------------------------- 
    let test_9mers=unique_9_mers
                                                    .choose_multiple(&mut rng,num_test_examples)
                                                    .map(|(mer,peptides)|(mer.clone(),peptides.clone()))
                                                    .collect::<Vec<_>>();
    let train_9mers=unique_9_mers
                                                    .iter()
                                                    .filter(|(mer,peptides)| 
                                                            {
                                                                for (test_mer, _) in test_9mers.iter()
                                                                {
                                                                    if mer ==test_mer{return false}
                                                                }
                                                                true
                                                            }
                                                        )
                                                    .map(|(mer,peptides)|(mer.clone(),peptides.clone()))
                                                    .collect::<Vec<_>>();
    // Get train and test peptides
    //----------------------------
    let test_peptides=test_9mers
                        .into_iter()
                        .map(|(mer,peptides)| peptides)
                        .flatten()
                        .collect::<Vec<_>>();
    
    let train_peptides=train_9mers
                        .into_iter()
                        .map(|(mer,peptides)| peptides)
                        .flatten()
                        .collect::<Vec<_>>();
    
    // Generate train and test dataset using proteome sampling 
    //-------------------------------------------------------
    // Prepare and sample the dataset 
    //-------------------------------
    let test_database=generate_negative_by_sampling_rs(test_peptides,&target_proteome,fold_neg);
    let train_database=generate_negative_by_sampling_rs(train_peptides,&target_proteome,fold_neg); 
    
    // return the results 
    //-------------------
    (train_database,test_database)
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
        (encoded_test_seq,encoded_test_labels),
        (encoded_train_seq,encoded_train_labels)
    )
}

/// ### Summary
/// The working engine for generating training datasets through same protein sampling 
/// ### Parameters
/// positive_examples: List of string representing positive peptides 
/// fold_neg: The fold of negative examples an integer representing the ration of positives to negative, if 1 then 1 negative is generated for every positive
///     meanwhile if it is set to 10 then 10 negative examples are generated from every positive examples.  
/// max_len: The maximum length to transfer a variable length peptides into a fixed length peptides. Here, shorter peptides are zero padded into this predefined 
/// sequence length. while longer peptides are trimmed into this length. 
/// test_size: The size of the test dataset, a float in the range (0,1)
/// ## Returns 
/// A tuple of two tuples with the following structure
///                         train_tuple:
///                               |---> train_seq:    A vector of string containing the training peptide sequences
///                               |---> train_label:  A vector of u8 containing or representing train label
///                         test_tuple:
///                               |---> test_seq: A vector of string containing the testing peptide sequences
///                               |---> test_label:   A vector of u8 containing or representing test label
fn prepare_train_ds_same_protein_sampling(positive_examples:&Vec<String>,proteome:&HashMap<String,String>,
        fold_neg:u32,test_size:f32)->((Vec<String>, Vec<u8>),(Vec<String>, Vec<u8>))
{
    // check the input is correct
    if positive_examples.len()==0{panic!("Input collection of positive examples is empty");}
    if test_size >=1.0 || test_size<=0.0 {panic!("your test size: {} is out on range, it must be a value between [0,1)",test_size)}

    // create the number of examples
    let num_test_examples=(test_size*positive_examples.len() as f32 ) as usize;
    let num_train_examples= positive_examples.len() as usize - num_test_examples;
    // first let's group by 9 mers
    let unique_9_mers=group_by_9mers_rs(positive_examples)
            .into_iter()
            .collect::<Vec<_>>();
    
    // get parent proteins and remove bound peptides
    //----------------------------------------------
    let positive_parent=group_peptides_by_parent_rs(positive_examples, proteome)
        .into_iter()        
        .map(|(pep,parents)|parents)
        .flatten()
        .collect::<Vec<_>>(); 

    // filter the database from positive proteins
    //-------------------------------------------
    let target_proteome=proteome
        .into_iter()
        .filter(|(name,seq)| positive_parent.contains(name))
        .collect::<HashMap<_,_>>(); 


    // create a thread for splitting the dataset 
    let mut rng=rand::thread_rng(); 

    // Split the data into test and train
    //------------------------------- 
    let test_9mers=unique_9_mers
                                                    .choose_multiple(&mut rng,num_test_examples)
                                                    .map(|(mer,peptides)|(mer.clone(),peptides.clone()))
                                                    .collect::<Vec<_>>();
    let train_9mers=unique_9_mers
                                                    .iter()
                                                    .filter(|(mer,peptides)| 
                                                            {
                                                                for (test_mer, _) in test_9mers.iter()
                                                                {
                                                                    if mer ==test_mer{return false}
                                                                }
                                                                true
                                                            }
                                                        )
                                                    .map(|(mer,peptides)|(mer.clone(),peptides.clone()))
                                                    .collect::<Vec<_>>();
    // Get train and test peptides
    //----------------------------
    let test_peptides=test_9mers
                        .into_iter()
                        .map(|(mer,peptides)| peptides)
                        .flatten()
                        .collect::<Vec<_>>();
    
    let train_peptides=train_9mers
                        .into_iter()
                        .map(|(mer,peptides)| peptides)
                        .flatten()
                        .collect::<Vec<_>>();
    
    // Generate train and test dataset using proteome sampling 
    //-------------------------------------------------------
    // Prepare and sample the dataset 
    //-------------------------------
    let test_database=generate_negative_by_sampling_rs(test_peptides,&target_proteome,fold_neg);
    let train_database=generate_negative_by_sampling_rs(train_peptides,&target_proteome,fold_neg); 
    
    // return the results 
    //-------------------
    (train_database,test_database)
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
    let(train_database,test_database)=prepare_train_ds_same_protein_sampling(&positive_examples,fold_neg,test_size);
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


/// ### Summary
/// The working engine for generating training datasets through sampling from proteins that expressed in a specific tissue 
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
/// ## Returns 
/// A tuple of two tuples with the following structure
///                         train_tuple:
///                               |---> train_seq:    A vector of string containing the training peptide sequences
///                               |---> train_label:  A vector of u8 containing or representing train label
///                         test_tuple:
///                               |---> test_seq: A vector of string containing the testing peptide sequences
///                               |---> test_label:   A vector of u8 containing or representing test label
fn prepare_train_ds_expressed_protein_sampling(positive_examples:Vec<String>,proteome:HashMap<String,String>,
    path2expression_map:&Path, tissue_name:String, threshold:f32,
    fold_neg:u32,test_size:f32)->((Vec<String>, Vec<u8>),(Vec<String>, Vec<u8>))
{
    
     // check the input is correct
     if positive_examples.len()==0{panic!("Input collection of positive examples is empty");}
     if test_size >=1.0 || test_size<=0.0 {panic!("your test size: {} is out on range, it must be a value between [0,1)",test_size)}
 
     let expression_db=ExpressionTable::read_tsv(path2expression_map, None).unwrap().to_hashmap_parallel();
     let target_tissue_exp_db=match expression_db.get(&tissue_name)
     {
         Some(table)=>table,
         None=>panic!(format!("The provided tissue name:{} does not exist in the database.",tissue_name))
     }; 
     // get a list of expressed proteins 
     //---------------------------------
     let target_proteins=target_tissue_exp_db
             .iter()
             .filter(|(protein_name,exp_level)|exp_level>threshold)
             .map(|(protein_name,_)|protein_name)
             .collect::<Vec<_>>();
     
     // filter the list of proteomes
     //-----------------------------
     let target_proteome=proteome
         .iter()
         .filter(|(protein_name,seq)|target_proteins.contains(protein_name))
         .map(|(protein_name,seq)|(protein_name.clone(),seq.clone()))
         .collect::<HashMap<_,_>>(); 
 
     // create the number of examples
     let num_test_examples=(test_size*positive_examples.len() as f32 ) as usize;
     let num_train_examples= positive_examples.len() as usize - num_test_examples;
     // first let's group by 9 mers
     let unique_9_mers=group_by_9mers_rs(positive_examples)
             .into_iter()
             .collect::<Vec<_>>();
     
     // get parent proteins and remove bound peptides
     //----------------------------------------------
     let positive_parent=group_peptides_by_parent_rs(positive_examples, &proteome)
         .into_iter()        
         .map(|(pep,parents)|parents)
         .flatten()
         .collect::<Vec<_>>(); 
 
     // filter the database from positive proteins
     //-------------------------------------------
     let target_proteome=proteome
         .into_iter()
         .filter(|(name,seq)| positive_parent.contains(name))
         .collect::<HashMap<_,_>>(); 
 
 
     // create a thread for splitting the dataset 
     let mut rng=rand::thread_rng(); 
 
     // Split the data into test and train
     //------------------------------- 
     let test_9mers=unique_9_mers
                                                     .choose_multiple(&mut rng,num_test_examples)
                                                     .map(|(mer,peptides)|(mer.clone(),peptides.clone()))
                                                     .collect::<Vec<_>>();
     let train_9mers=unique_9_mers
                                                     .iter()
                                                     .filter(|(mer,peptides)| 
                                                             {
                                                                 for (test_mer, _) in test_9mers.iter()
                                                                 {
                                                                     if mer ==test_mer{return false}
                                                                 }
                                                                 true
                                                             }
                                                         )
                                                     .map(|(mer,peptides)|(mer.clone(),peptides.clone()))
                                                     .collect::<Vec<_>>();
     // Get train and test peptides
     //----------------------------
     let test_peptides=test_9mers
                         .into_iter()
                         .map(|(mer,peptides)| peptides)
                         .flatten()
                         .collect::<Vec<_>>();
     
     let train_peptides=train_9mers
                         .into_iter()
                         .map(|(mer,peptides)| peptides)
                         .flatten()
                         .collect::<Vec<_>>();
     
     // Generate train and test dataset using proteome sampling 
     //-------------------------------------------------------
     // Prepare and sample the dataset 
     //-------------------------------
     let test_database=generate_negative_by_sampling_rs(test_peptides,&target_proteome,fold_neg);
     let train_database=generate_negative_by_sampling_rs(train_peptides,&target_proteome,fold_neg); 
    
    // return the results 
    //-------------------
    (train_database,test_database)
}

/// ### Signature(positive_examples:List[str],proteome:Dict[str,str],path2expression_map:str,
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
            path2expression_map:&Path, tissue_name:String, threshold:f32,
            fold_neg:u32,max_len:usize,test_size:f32)->(
                (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>>),
                (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>>)
            )
{
    // prepare the train and test datasets through sampling from the same protein 
    let(train_database,test_database)=prepare_train_ds_expressed_protein_sampling(&positive_examples,proteome,
        path2expression_map,tissue_name,
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
            path2gene_expression:Option<Path>,
            tissue_name:Option<String>,
            path2pseudoSeq:&Path,
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
    match method
    {
        "shuffle"=>(),
        "prot_sample"=>
        {
            match proteome
            {
                Some(proteome)=>(),
                None=>panic!("Method: prot_sample can not be used when the input proteome has not been provided")
            }
        }, 
        "same"=>
        {
            match proteome
            {
                Some(proteome)=>(),
                None=>panic!("Method: same can not be used when the input proteome has not been provided")
            }
        },
        "exp_samp"=>
        {
            match proteome
            {
                Some(proteome)=>(),
                None=>panic!("Method: same can not be used when the input proteome has not been provided")
            }
            match path2gene_expression
            {
                Some(path)=>(),
                None=>panic!("Method: exp_samp can not be used when the input expression table has not been provided")
            }
            match tissue_name
            {
                Some(path)=>(),
                None=>panic!("Method: exp_samp can not be used when the input tissue name has not been provided")
            }
        }
        _=>panic!(format!("Method: {} is currently not supported, only: random, prot_sample, same or exp_samp are supported",method))
    };
    // Read the pseudo-sequences
    let pseudo_seq=read_pseudo_seq(path2pseudoSeq); 

    // Count the number of entries in the HashMap to generate the results
    //-------------------------------------------------------------------
    let num_entries=fold_neg*positive_examples
                .iter()
                .map(|(alleles,peptides)| if peptides.len()>100{return 50;} else {return 0;})
                .sum::<u32>(); 
    
    // Compute the average number of test and size examples
    //-----------------------------------------------------
    let num_test_examples=(test_size*num_entries as f32) as u32;
    let num_train_examples=num_entries-num_test_examples;
    
    // allocate vectors to hold the results
    //-------------------------------------
    let (mut train_labels,mut train_alleles,mut train_sequences)=(Vec::with_capacity(num_train_examples),
                    Vec::with_capacity(num_train_examples),Vec::with_capacity(num_train_examples));   

    let (mut test_labels,mut test_alleles,mut test_sequences)=(Vec::with_capacity(num_test_examples),
    Vec::with_capacity(num_test_examples),Vec::with_capacity(num_test_examples));   

    // loop over alleles and generate 
    //-------------------------------
    for (allele_name,peptides) in positive_examples.iter()
    {
        if peptides.len()>50{continue;}// check that the number of peptides is above 50 and skip it otherwise

        let ((mut train_seq,mut train_label),(mut test_seq,mut test_label))=match method
        {
            "shuffle"=>prepare_train_ds_shuffling(peptides,fold_neg,test_size),
            "prot_sample"=>prepare_train_ds_proteome_sampling(peptides,&proteome.unwrap(),fold_neg,test_size), 
            "same"=>prepare_train_ds_same_protein_sampling(peptides,&proteome.unwrap(),fold_neg,test_size),
            "exp_samp"=>prepare_train_ds_expressed_protein_sampling(peptides,&proteome.unwrap(),
                            &path2expression_map.unwrap(),tissue_name.unwrap(),exp_threshold.unwrap(),
                            fold_neg,test_size),
            _=>panic!("Un supported methods")
        };

        // Extend the allocated vectors
        // extend the training arrays
        //---------------------------
        train_alleles.append(&mut vec![pseudo_seq.get(allele_name).unwrap();train_seq.len()]);
        train_sequences.append(&mut train_seq); 
        train_labels.append(&mut train_label);
        // extend the test arrays
        //-----------------------
        test_alleles.append(&mut vec![pseudo_seq.get(allele_name).unwrap();test_seq.len()]);
        test_sequences.append(&mut test_seq); 
        test_labels.append(&mut test_label);
    }
    // Numerically encode the datasets
    //--------------------------------
    let encoded_train_labels= Array::from_shape_vec((train_sequences.len(),1), train_labels).unwrap().to_pyarray(py);
    let encoded_train_peptide_seq=encode_sequence_rs(train_sequences,max_len).to_pyarray(py);
    let encoded_train_pseudo_seq=encode_sequence_rs(train_alleles,max_len).to_pyarray(py);

    let encoded_test_labels= Array::from_shape_vec((test_sequences.len(),1), test_labels).unwrap().to_pyarray(py);
    let encoded_test_peptide_seq=encode_sequence_rs(test_sequences,max_len).to_pyarray(py);
    let encoded_test_pseudo_seq=encode_sequence_rs(test_alleles,max_len).to_pyarray(py);  
    
    // return the results 
    //-------------------
    (
        (encoded_train_peptide_seq,encoded_train_pseudo_seq,encoded_train_labels),
        (encoded_test_peptide_seq,encoded_test_pseudo_seq,encoded_test_labels)
    )
}

// CREATE THE FUNCTIONS
//#[pyfunction]
//pub fn generate_train_ds_pm_ipQ<'py>
//pub fn generate_train_ds_pm_ipQMA<'py>
