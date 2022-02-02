use std::collections::HashMap;

/// The module contains a collection of function that can be used for training sequence only models 
/// 
/// 
/// 

use ndarray::{Dim, Array};
use numpy::{PyArray, ToPyArray};
use pyo3::{pyfunction,Python};
use crate::peptides::{group_by_9mers_rs,generate_a_train_db_by_shuffling_rs,encode_sequence_rs, group_peptides_by_parent_rs,generate_negative_by_sampling_rs}; 
use rand;
use rand::seq::SliceRandom;


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
///                               |---> encoded_train_label:  a tensor with shape(num_train_examples,1) and type u8
/// -----------------------------------------------
#[pyfunction]
pub fn generate_train_ds_shuffling_sm<'py>(py:Python<'py>, 
            positive_examples:Vec<String>,fold_neg:u32,max_len:usize,test_size:f32)->(
                (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>>),
                (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>>)
            )
{
    // check the input is correct
    if positive_examples.len()==0{panic!("Input collection of positive examples is empty");}
    if test_size >=1.0 || test_size<=0.0 {panic!("your test size: {} is out on range, it must be a value between [0,1)",test_size)}

    // create the number of examples
    let num_test_example=(test_size*positive_examples.len() as f32 ) as usize;
    let num_train_examples= positive_examples.len() as usize - num_test_example;
    // first let's group by 9 mers
    let unique_9_mers=group_by_9mers_rs(positive_examples)
            .into_iter()
            .collect::<Vec<_>>();
            
    // create a thread for splitting the dataset 
    let mut rng=rand::thread_rng(); 

    // Split the data into test and train
    //------------------------------- 
    let test_9mers=unique_9_mers
                                                    .choose_multiple(&mut rng,num_test_example)
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
///                               |---> encoded_train_label:  a tensor with shape(num_train_examples,1) and type u8
/// -----------------------------------------------
#[pyfunction]
pub fn generate_train_ds_proteome_sampling_sm<'py>(py:Python<'py>, 
            positive_examples:Vec<String>,proteome:HashMap<String,String>,fold_neg:u32,max_len:usize,test_size:f32)->(
                (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>>),
                (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>>)
            )
{
    // check the input is correct
    if positive_examples.len()==0{panic!("Input collection of positive examples is empty");}
    if test_size >=1.0 || test_size<=0.0 {panic!("your test size: {} is out on range, it must be a value between [0,1)",test_size)}

    // create the number of examples
    let num_test_example=(test_size*positive_examples.len() as f32 ) as usize;
    let num_train_examples= positive_examples.len() as usize - num_test_example;
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
                                                    .choose_multiple(&mut rng,num_test_example)
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
//fn prepare_train_ds_same_transcript_sampling()

//fn prepare_train_ds_expressed_protein_sampling()