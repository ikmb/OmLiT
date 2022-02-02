/// The module contains a collection of function that can be used for training sequence only models 
/// 
/// 
/// 

use ndarray::{Dim, Array};
use numpy::{PyArray, ToPyArray};
use pyo3::{pyfunction,Python};
use crate::peptides::{group_by_9mers_rs,generate_a_train_db_by_shuffling_rs,encode_sequence_rs}; 
use rand;
use rand::seq::SliceRandom;



#[pyfunction]
pub fn prepare_train_ds_shuffling<'py>(py:Python<'py>, 
            positive_examples:Vec<String>,fold_neg:u32,max_len:usize,test_size:f32)->(
                (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>>),
                (&'py PyArray<u8,Dim<[usize;2]>>, &'py PyArray<u8,Dim<[usize;2]>>)
            )
{
    // check the input is correct
    if positive_examples.len()==0{panic!("Input collection of positive examples is empty");}
    if test_size >=1.0 || test_size<0.0 {panic!("your test size: {} is out on range, it must be a value between [0,1)",test_size)}

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

//fn prepare_train_ds_proteome_sampling()

//fn prepare_train_ds_same_transcript_sampling()

//fn prepare_train_ds_expressed_protein_sampling()