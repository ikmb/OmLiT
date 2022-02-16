use std::{collections::HashMap, path::Path};
use crate::{
        peptides::{generate_a_train_db_by_shuffling_rs, generate_negative_by_sampling_rs, group_peptides_by_parent_rs},
         utils::{
            split_positive_examples_into_test_and_train, clean_data_sets},
            functions::read_cashed_db};

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

    // prepare the train and test positive examples                                           
    let (train_seq,test_seq)=split_positive_examples_into_test_and_train(positive_examples,test_size); 
    
    // Prepare and sample the dataset 
    //-------------------------------
    let test_database=generate_a_train_db_by_shuffling_rs(train_seq,fold_neg);
    let train_database=generate_a_train_db_by_shuffling_rs(test_seq,fold_neg); 
    
    // clean the results from overlaps 
    //-------------------
    let (train_database,test_database)=clean_data_sets(train_database,test_database); 
    
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

    // generate the list for sampling the negative peptides
    //-----------------------------------------------------
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

    // prepare the train and test positive examples                                           
    let (train_seq,test_seq)=split_positive_examples_into_test_and_train(positive_examples,test_size); 

    // Prepare and sample the dataset 
    //-------------------------------
    let test_database=generate_negative_by_sampling_rs(train_seq,&target_proteome,fold_neg);
    let train_database=generate_negative_by_sampling_rs(test_seq,&target_proteome,fold_neg); 

    // clean the results from overlaps 
    //-------------------
    let (train_database,test_database)=clean_data_sets(train_database,test_database); 
    
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

    // compute the target proteome 
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

    // prepare the train and test positive examples                                           
    let (train_seq,test_seq)=split_positive_examples_into_test_and_train(positive_examples,test_size); 
   
    // Generate train and test dataset using proteome sampling 
    //-------------------------------------------------------
    // Prepare and sample the dataset 
    //-------------------------------
    let test_database=generate_negative_by_sampling_rs(test_seq,&target_proteome,fold_neg);
    let train_database=generate_negative_by_sampling_rs(train_seq,&target_proteome,fold_neg); 

    // clean the results from overlaps 
    //-------------------
    let (train_database,test_database)=clean_data_sets(train_database,test_database); 

    // return the results 
    //-------------------
    (train_database,test_database)
}


/// ### Summary
/// The working engine for generating training datasets through sampling from proteins that expressed in a specific tissue 
/// ### Parameters
/// positive_examples: List of string representing positive peptides 
/// proteome: a dict of protein name and protein sequences
/// path2cashed_db: The path to load a cahsed database object 
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
    path2cashed_db:&Path, tissue_name:&String, threshold:f32,
    fold_neg:u32,test_size:f32)->((Vec<String>, Vec<u8>),(Vec<String>, Vec<u8>))
{
    
    // check the input is correct
    if positive_examples.len()==0{panic!("Input collection of positive examples is empty");}
    if test_size >=1.0 || test_size<=0.0 {panic!("your test size: {} is out on range, it must be a value between [0,1)",test_size)}

    let expression_db=read_cashed_db(path2cashed_db); 
    let target_tissue_exp_db=match expression_db.get(tissue_name)
    {
        Some(table)=>table,
        None=>panic!("The provided tissue name:{} does not exist in the database.",tissue_name)
    }; 
    // get a list of expressed proteins 
    //---------------------------------
    let target_proteins=target_tissue_exp_db
            .iter()
            .filter(|(_,prot_info)|prot_info.get_expression() > threshold)
            .map(|(protein_name,_)|protein_name.clone())
            .collect::<Vec<_>>();
    
    // filter the list of proteomes
    //-----------------------------
    let exp_proteome=proteome
        .iter()
        .filter(|(protein_name,_)|target_proteins.contains(protein_name))
        .map(|(protein_name,seq)|(protein_name.clone(),seq.clone()))
        .collect::<HashMap<_,_>>(); 
    
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

    // prepare the train and test positive examples                                           
    let (train_seq,test_seq)=split_positive_examples_into_test_and_train(positive_examples,test_size); 
     
    // Generate train and test dataset using proteome sampling 
    //-------------------------------------------------------
    // Prepare and sample the dataset 
    //-------------------------------
    let test_database=generate_negative_by_sampling_rs(test_seq,&target_proteome,fold_neg);
    let train_database=generate_negative_by_sampling_rs(train_seq,&target_proteome,fold_neg); 
    
    // clean the results from overlaps 
    //-------------------
    let (train_database,test_database)=clean_data_sets(train_database,test_database); 
    // return the results 
    //-------------------
    (train_database,test_database)
}