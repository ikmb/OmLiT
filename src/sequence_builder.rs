use std::{collections::{HashMap, HashSet}, path::Path, ops::RangeBounds};

use rand::prelude::{SliceRandom, IteratorRandom};
use rayon::iter::{IntoParallelIterator, ParallelIterator, IntoParallelRefIterator};

use crate::{peptides::{group_by_9mers_rs, generate_a_train_db_by_shuffling_rs, generate_negative_by_sampling_rs, group_peptides_by_parent_rs, fragment_peptide_into_9_mers}, geneExpressionIO::ExpressionTable};

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
pub fn prepare_train_ds_shuffling(positive_examples:&Vec<String>,fold_neg:u32,test_size:f32)->((Vec<String>, Vec<u8>),(Vec<String>, Vec<u8>))
{
    
    // check the input is correct
    if positive_examples.len()==0{panic!("Input collection of positive examples is empty");}
    if test_size >=1.0 || test_size<=0.0 {panic!("your test size: {} is out on range, it must be a value between [0,1)",test_size)}

    let num_test_examples=(positive_examples.len() as f32 *test_size) as usize; 
    
    // create a thread for splitting the dataset 
    let mut rng=rand::thread_rng(); 

    // Split the data into test and train
    //------------------------------- 
    let index_test=(0..positive_examples.len())
                                    .choose_multiple(&mut rng,num_test_examples); 
    
    let test_seq = index_test
                                        .iter() 
                                        .map(|index|positive_examples[*index].clone())
                                        .collect::<Vec<_>>();

    let mut train_seq=(0..positive_examples.len())
                        .filter(|index|!index_test.contains(index))
                        .map(|index|positive_examples[index].clone())
                        .collect::<Vec<_>>();
    // Get train and test peptides
    //----------------------------
    // Extract the test examples that are similar to the train dataset.
    let mut similar_to_train=test_seq
                    .par_iter()
                    .filter(|peptide|
                    {
                        for target_test_mers in fragment_peptide_into_9_mers(peptide)
                        {
                            for train_pep in train_seq.iter(){if train_pep.contains(&target_test_mers){return true}} 
                        }
                        return false
                    })
                    .map(|elem|elem.clone())
                    .collect::<Vec<_>>(); 
    // Extract the dataset from the test dataset 
    //------------------------------------------
    let filtered_test_seq=test_seq
                        .into_par_iter()
                        .filter(|peptide|!similar_to_train.contains(&peptide))
                        .collect::<Vec<_>>(); 
                        
    train_seq.append(&mut similar_to_train);                                           

    // Prepare and sample the dataset 
    //-------------------------------
    let test_database=generate_a_train_db_by_shuffling_rs(train_seq,fold_neg);
    let train_database=generate_a_train_db_by_shuffling_rs(filtered_test_seq,fold_neg); 
    
    // return the results 
    //-------------------
    (train_database,test_database)
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
pub fn prepare_train_ds_proteome_sampling(positive_examples:&Vec<String>,proteome:&HashMap<String,String>,
    fold_neg:u32,test_size:f32)->((Vec<String>, Vec<u8>),(Vec<String>, Vec<u8>))
{

    // check the input is correct
    if positive_examples.len()==0{panic!("Input collection of positive examples is empty");}
    if test_size >=1.0 || test_size<=0.0 {panic!("your test size: {} is out on range, it must be a value between [0,1)",test_size)}

    // create the number of examples
    let num_test_examples=(test_size*positive_examples.len() as f32 ) as usize;
    // first let's group by 9 mers
    let unique_9_mers=group_by_9mers_rs(positive_examples)
            .into_iter()
            .collect::<Vec<_>>();

    // get parent proteins and remove bound peptides
    //----------------------------------------------
    let positive_parent=group_peptides_by_parent_rs(positive_examples, &proteome)
        .into_iter()        
        .map(|(_,parents)|parents)
        .flatten()
        .collect::<Vec<_>>(); 

    // filter the database from positive proteins
    //-------------------------------------------
    let target_proteome=proteome
        .into_iter()
        .filter(|(name,_)| !positive_parent.contains(name))
        .map(|(name,seq)|(name.clone(),seq.clone()))
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
                                                    .filter(|(mer,_)| 
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
                        .map(|(_,peptides)| peptides)
                        .flatten()
                        .collect::<Vec<_>>();

    let train_peptides=train_9mers
                        .into_iter()
                        .map(|(_,peptides)| peptides)
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
pub fn prepare_train_ds_same_protein_sampling(positive_examples:&Vec<String>,proteome:&HashMap<String,String>,
    fold_neg:u32,test_size:f32)->((Vec<String>, Vec<u8>),(Vec<String>, Vec<u8>))
{
    // check the input is correct
    if positive_examples.len()==0{panic!("Input collection of positive examples is empty");}
    if test_size >=1.0 || test_size<=0.0 {panic!("your test size: {} is out on range, it must be a value between [0,1)",test_size)}

    // create the number of examples
    let num_test_examples=(test_size*positive_examples.len() as f32 ) as usize;
    // first let's group by 9 mers
    let unique_9_mers=group_by_9mers_rs(positive_examples)
            .into_iter()
            .collect::<Vec<_>>();

    // get parent proteins and remove bound peptides
    //----------------------------------------------
    let positive_parent=group_peptides_by_parent_rs(positive_examples, proteome)
        .into_iter()        
        .map(|(_,parents)|parents)
        .flatten()
        .collect::<Vec<_>>(); 

    // filter the database from positive proteins
    //-------------------------------------------
    let target_proteome=proteome
        .into_iter()
        .filter(|(name,_)| positive_parent.contains(name))
        .map(|(name,seq)|(name.clone(),seq.clone()))
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
                                                    .filter(|(mer,_)| 
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
                        .map(|(_,peptides)| peptides)
                        .flatten()
                        .collect::<Vec<_>>();

    let train_peptides=train_9mers
                        .into_iter()
                        .map(|(_,peptides)| peptides)
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
pub fn prepare_train_ds_expressed_protein_sampling(positive_examples:&Vec<String>,proteome:&HashMap<String,String>,
    path2expression_map:&Path, tissue_name:&String, threshold:f32,
    fold_neg:u32,test_size:f32)->((Vec<String>, Vec<u8>),(Vec<String>, Vec<u8>))
{
    
     // check the input is correct
     if positive_examples.len()==0{panic!("Input collection of positive examples is empty");}
     if test_size >=1.0 || test_size<=0.0 {panic!("your test size: {} is out on range, it must be a value between [0,1)",test_size)}
 
     let expression_db=ExpressionTable::read_tsv(path2expression_map, None).unwrap().to_hashmap_parallel();
     let target_tissue_exp_db=match expression_db.get(tissue_name)
     {
         Some(table)=>table,
         None=>panic!("The provided tissue name:{} does not exist in the database.",tissue_name)
     }; 
     // get a list of expressed proteins 
     //---------------------------------
     let target_proteins=target_tissue_exp_db
            .iter()
            .filter(|(_,exp_level)|exp_level>&&threshold)
            .map(|(protein_name,_)|protein_name.clone())
            .collect::<Vec<_>>();
     
     // filter the list of proteomes
     //-----------------------------
     let exp_proteome=proteome
         .iter()
         .filter(|(protein_name,_)|target_proteins.contains(protein_name))
         .map(|(protein_name,seq)|(protein_name.clone(),seq.clone()))
         .collect::<HashMap<_,_>>(); 
 
     // create the number of examples
     let num_test_examples=(test_size*positive_examples.len() as f32 ) as usize;
     // first let's group by 9 mers
     let unique_9_mers=group_by_9mers_rs(positive_examples)
             .into_iter()
             .collect::<Vec<_>>();
     
     // get parent proteins and remove bound peptides
     //----------------------------------------------
     let positive_parent=group_peptides_by_parent_rs(positive_examples, &proteome)
         .into_iter()        
         .map(|(_,parents)|parents)
         .flatten()
         .collect::<Vec<_>>(); 
 
     // filter the database from positive proteins
     //-------------------------------------------
     let target_proteome=exp_proteome
         .into_iter()
         .filter(|(name,_)| !positive_parent.contains(name))
         .map(|(name,seq)|(name.clone(),seq.clone()))
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
                                                     .filter(|(mer,_)| 
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
                         .map(|(_,peptides)| peptides)
                         .flatten()
                         .collect::<Vec<_>>();
     
     let train_peptides=train_9mers
                         .into_iter()
                         .map(|(_,peptides)| peptides)
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