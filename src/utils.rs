use std::{collections::{HashMap, HashSet}, path::Path, cmp::Ordering};
use csv; 
use rand::prelude::IteratorRandom;
use rayon::prelude::*;

use crate::{peptides::{group_by_9mers_rs, fragment_peptide_into_9_mers}, peptides::sample_a_negative_peptide, protein_info::ProteinInfo, functions::read_cashed_db, constants::UNIQUE_GO_TERMS}; 
/// ### Summary 
/// A reader function for reading HLA pseudo sequences
/// ### Parameters 
/// The path to the file containing the pseudo sequences
/// ### Return
/// returns a hashmap of allele names to protein sequences
/// ---------------------------
pub fn read_pseudo_seq(input2file:&Path)->HashMap<String,String>
{
    let mut file_reader=csv::ReaderBuilder::new()
                        .has_headers(false)
                        .delimiter(b'\t')
                        .from_path(input2file)
                        .unwrap(); 
    let mut res_map=HashMap::new();

    for record in file_reader.records()
    {
        let row=record.unwrap();
        res_map.insert(row[0].to_string(), row[1].to_string());
    }
    res_map
}

/// ### Summary 
/// The function is used to generate a dataset of positive and negative from a collection of positive peptides. 
/// ### Parameter 
///  group_by_alleles: A nested hashmap with the following structure:
///                                                     Allele name:
///                                                               |__ Tissue name (Key) --> peptides <Vec<String>> A vector of peptides sequences (Values). 
///  target_proteomes: A nested HashMap representing the negative proteins, from which negative sampling shall take place (i.e. negative sampling )
///  fold_neg: The amount of negative examples to sample, Xfold the training datasets, where 1 represent a ratio of 1 negative per each positive example while 5 represents  
///  5 negatives from each positive example. 
///  test_size: The size of the training dataset, a fraction in the interval (0,1)
///  proteome: the reference proteome, which is a hashmap of protein names linked with protein sequences
/// 
/// ### Return 
/// The function returns a tuple of two tuples representing the training and the test dataset, each structure has the following structure
///     a. A vector of strings representing peptide sequence 
///     b. A vector of strings representing allele names 
///     c. A vector of strings representing tissue names            
///     d. A vector of integers (u8) representing the labels of each peptide, 1 represent binder and 0 represent non-binders 
pub fn sample_negatives_from_positive_data_structure(group_by_alleles:HashMap<String,HashMap<String,Vec<String>>>,
    target_proteomes:&HashMap<String,Vec<String>>, fold_neg:u32, test_size:f32,proteome:&HashMap<String,String>
    )->((Vec<String>/* peptide */,Vec<String>/* allele name */,Vec<String>/* tissue name */,Vec<u8> /* label*/),
        (Vec<String>/* peptide */,Vec<String>/* allele name */,Vec<String>/* tissue name */,Vec<u8> /* label*/)
    )
{
    // Check the test size is valid 
    if test_size<=0.0 || test_size>=1.0{panic!("In-valid test size: expected the test-size to be a float bigger than 0.0 and smaller than 1.0, however, the test size is: {}",test_size);}
    // Compute the size of the test dataset 
    //-------------------------------------
    let results:HashMap<String,HashMap<_,_>>=group_by_alleles
        .into_par_iter()
        .map(|(allele_name,tissue_to_peptides)|
        {
            let results_from_all_tissues=tissue_to_peptides
                .into_iter()
                .map(|(tissue_name,peptides)|
                {
                    
                    // group by the 9 mers cores
                    //-----------------------------------------------------
                    let grouped_by_peptides=group_by_9mers_rs(&peptides);
                    let peptide_mers=grouped_by_peptides
                        .keys()
                        .map(|mers_core|mers_core.to_string())
                        .collect::<Vec<_>>(); 
                    
                    // Create the target tissue proteome to sample from 
                    //-------------------------------------------------
                    let target_proteins=target_proteomes.get(&tissue_name).unwrap();
                    let target_protein_seq=proteome
                            .iter()
                            .filter(|(name,_)|target_proteins.contains(name))
                            .map(|(name,seq)|(name.clone().to_owned(),seq.clone().to_owned()))
                            .collect::<Vec<_>>();
                    
                    // Compute the size of the test dataset 
                    //-------------------------------------
                    let num_dp=grouped_by_peptides.len();
                    let num_test_dp=(num_dp as f32 *test_size) as usize; 
                    // create a random number generator
                    //--------------------- 
                    let mut rng=rand::thread_rng(); 
                    // prepare the train and test indices
                    //-----------------------------------
                    let test_index=(0..grouped_by_peptides.len())
                        .choose_multiple(&mut rng, num_test_dp);
                    
                    let train_index=(0..grouped_by_peptides.len())
                            .filter(|index|!test_index.contains(index))
                            .collect::<Vec<_>>(); 
                
                    // prepare the train dataset
                    //--------------------------
                    // 1. positive peptides 
                    let mut positive_train_pep=train_index
                        .into_iter()
                        .map(|index|grouped_by_peptides.get(&peptide_mers[index]).unwrap())
                        .flatten()
                        .map(|pep|pep.clone().to_owned())
                        .collect::<Vec<_>>();
                    let mut positive_train_label=vec![1;positive_train_pep.len()]; 
                    
                    // 2. negative peptides
                    let mut sampled_negatives_train_pep=(0..(positive_train_pep.len() as u32 * fold_neg))
                        .into_iter()
                        .map(|_|sample_a_negative_peptide(&peptides,&target_protein_seq))
                        .collect::<Vec<String>>();
                    let mut sampled_negatives_train_label=vec![0;sampled_negatives_train_pep.len()]; 
                    
                    // 3. combine the results into a two vectors from sequences and labels
                    let mut train_seq=Vec::with_capacity(positive_train_pep.len()+sampled_negatives_train_pep.len()); 
                    train_seq.append(&mut positive_train_pep); 
                    train_seq.append(&mut sampled_negatives_train_pep); 

                    let mut train_labels=Vec::with_capacity(positive_train_label.len()+sampled_negatives_train_label.len());
                    train_labels.append(&mut positive_train_label); 
                    train_labels.append(&mut sampled_negatives_train_label);

                    // prepare the test dataset
                    //--------------------------
                    // 1. positive peptides 
                    let mut positive_test_pep=test_index
                        .into_iter()
                        .map(|index|grouped_by_peptides.get(&peptide_mers[index]).unwrap())
                        .flatten()
                        .map(|pep|pep.clone().to_owned())
                        .collect::<Vec<_>>();
                    let mut positive_test_label=vec![1;positive_test_pep.len()]; 
                    
                    // 2. negative peptides
                    let mut sampled_negatives_test_pep=(0..(positive_test_pep.len() as u32*fold_neg))
                        .into_iter()
                        .map(|_|sample_a_negative_peptide(&peptides,&target_protein_seq))
                        .collect::<Vec<String>>();
                    let mut sampled_negatives_test_label=vec![0;sampled_negatives_test_pep.len()]; 
                    
                    // 3. combine the results into a two vectors from sequences and labels
                    let mut test_seq=Vec::with_capacity(positive_test_pep.len()+sampled_negatives_test_pep.len()); 
                    test_seq.append(&mut positive_test_pep); 
                    test_seq.append(&mut sampled_negatives_test_pep); 

                    let mut test_labels=Vec::with_capacity(positive_test_label.len()+sampled_negatives_test_label.len());
                    test_labels.append(&mut positive_test_label); 
                    test_labels.append(&mut sampled_negatives_test_label);
                    
                    // generate and return the output 
                    (tissue_name,
                        ((train_seq,train_labels),
                        (test_seq,test_labels))
                    )
                })
                .collect::<HashMap<String,((Vec<String>,Vec<u8>),(Vec<String>,Vec<u8>))>>(); 
                (allele_name.to_string(),results_from_all_tissues)
        })
        .collect::<HashMap<String,HashMap<String,((Vec<String>,Vec<u8>),(Vec<String>,Vec<u8>))>>>();
        // Unroll the results in to the vector form 
        //------------------------------------------
        // 1. get number of train and test data points 
        let num_train_dp=results.iter()
            .map(|(_,tissue_data)|
            {
                tissue_data.iter().map(|(_,(train_data,_))|train_data.0.len()).sum::<usize>()
            }) 
            .sum::<usize>(); 
        let num_test_dp=results.iter()
        .map(|(_,tissue_data)|
        {
            tissue_data.iter().map(|(_,(_,test_data))|test_data.0.len()).sum::<usize>()
        }) 
        .sum::<usize>(); 

        // 2. allocate vectors to hold the results 
        let (mut train_peptide, mut train_allele_name, mut train_tissue_name, mut train_label)=(
            Vec::with_capacity(num_train_dp),Vec::with_capacity(num_train_dp),Vec::with_capacity(num_train_dp),
        Vec::with_capacity(num_train_dp)); 

        let (mut test_peptide, mut test_allele_name, mut test_tissue_name, mut test_label)=(
            Vec::with_capacity(num_test_dp),Vec::with_capacity(num_test_dp),Vec::with_capacity(num_test_dp),
        Vec::with_capacity(num_test_dp));  

        // 3. un-roll the data into vectors
        for (allele_name,allele_data) in  results
        {
            for (tissue_name, (mut train_data, mut test_data)) in allele_data
            {
                // Prepare the training data
                //-------------------------- 
                train_allele_name.append(&mut vec![allele_name.clone();train_data.0.len()]); 
                train_tissue_name.append(&mut vec![tissue_name.clone();train_data.0.len()]); 
                train_peptide.append(&mut train_data.0); 
                train_label.append(&mut train_data.1); 
                
                // Prepare the test data
                //----------------------
                test_allele_name.append(&mut vec![allele_name.clone();test_data.0.len()]); 
                test_tissue_name.append(&mut vec![tissue_name.clone();test_data.0.len()]); 
                test_peptide.append(&mut test_data.0); 
                test_label.append(&mut test_data.1); 
            }
        }
        // Return the results to the encoder 
        //----------------------------------
        (
            (train_peptide,train_allele_name,train_tissue_name,train_label),
            (test_peptide,test_allele_name,test_tissue_name,test_label)
        )
}

/// ### Summary 
/// Create a negative database for peptide sampling, the database is build as a hashmap with keys composite of the name of each tissue
/// and keys made of a vector of strings representing the ids of proteins that can be used for negative sampling from the corresponding tissue. 
/// ### Parameters
/// proteome: which is a hashmap of protein names linked with protein sequences
/// input2prepare: 
///     A tuple of three vectors representing the input to the functions:
///     1. Vector of strings representing the peptide sequences 
///     2. Vector of strings representing the allele names (current implementation represent a one allele per peptides)
///     3. Vector of strings representing the tissue name
/// anno_table: a hashmap representing the annotation table which is made of a nested hashmap with tissue names as keys and a hashmap containing protein id 
/// and protein information as values, i.e. a representation for the proteome state in different tissues. see the AnnotationTable for further details. 
/// threshold: A float representing the minimum gene expression level to considered a gene as expressed.    
/// ### Returns: 
///  A hashmap of tissue names along with a vector of protein ids which shall be used for generating the negatives. 
fn create_negative_database_from_positive(proteome:&HashMap<String,String>, 
    input2prepare:&(Vec<String>,Vec<String>,Vec<String>), 
    anno_table:&HashMap<String, HashMap<String, ProteinInfo>>,
    threshold:f32)->HashMap<String,Vec<String>>
{
    // Group the proteins by the tissues name using a pool of worker threads
    //-----------------------------------------------------------------------
    let unique_peptides=input2prepare.2.iter().collect::<HashSet<_>>(); 
    let positive_proteins_per_tissues=unique_peptides
                        .par_iter()
                        .map(|tissue_name|
                            {
                                // Get the peptides belonging to each tissue
                                //------------------------------------------ 
                                let peptides_from_tissue=input2prepare.0.iter().zip(input2prepare.1.iter())
                                        .filter(|(_,name_tissue)|tissue_name==name_tissue)
                                        .map(|(peptide,_)|peptide.to_string())
                                        .collect::<Vec<String>>();
                                // Get the parent protein of each peptide
                                //---------------------------------------
                                let parent_proteins=peptides_from_tissue
                                                    .iter()
                                                    .map(|pep|
                                                        {
                                                            proteome
                                                            .iter()
                                                            .map(|(prote_name,protein_seq)|
                                                            {
                                                                if protein_seq.contains(pep){prote_name.clone()}
                                                                else{"?".to_string()}
                                                            })
                                                            .collect::<Vec<String>>()
                                                        })
                                                        .flatten()
                                                        .filter(|pep| pep!="?")
                                                        .collect::<HashSet<String>>()
                                                        .into_iter()
                                                        .collect::<Vec<String>>();
                                (tissue_name.clone().to_owned(),parent_proteins)
                            }
                        )
                        .collect::<HashMap<_,_>>();
    // let's compute the negative per tissues
    //---------------------------------------
    positive_proteins_per_tissues
                .into_par_iter()
                .map(|(tissue_name,positive_proteins)|
                    {
                        match anno_table.get(&tissue_name)
                        {
                            Some(target_table)=>
                            {
                                let negative_proteins=target_table
                                                    .iter()
                                                    .filter(|(protein_name,protein_info)|protein_info.get_expression()>threshold)
                                                    .map(|(protein_name,_)|protein_name.clone())
                                                    .filter(|protein_name|!positive_proteins.contains(protein_name))
                                                    .collect::<Vec<_>>();
                                (tissue_name,negative_proteins)
                                //[(tissue_name,negative_proteins)].into_iter().collect::<HashMap::<_,_>>()
                            }
                            None=>(tissue_name,Vec::new())
                        }
                    })
                .collect::<HashMap<_,_>>()
} 
/// ### Summary
/// A wrapper functions that acts as a prelude for all functions in the train_mo (train multi omics) datasets  
/// ### Parameters 
/// input2prepare: A tuple of three vectors representing the input to the functions:
///     1. Vector of strings representing the peptide sequences 
///     2. Vector of strings representing the allele names (current implementation represent a one allele per peptides)
///     3. Vector of strings representing the tissue name
/// proteome: A hashmap of protein name and sequence
/// path2cashed_db: The path to load the cashed annotation table, see AnnotationTable for more details
/// path2pseudo_seq: The path to load the pseudo-sequences table for each allele
/// threshold: The gene expression threshold, i.e. the minimum expression to consider a protein as expressed, used for generating target proteomes, see for more details  
///  fold_neg: The amount of negative examples to sample, Xfold the training datasets, where 1 represent a ratio of 1 negative per each positive example while 5 represents  
///  5 negatives from each positive example. 
///  test_size: The size of the training dataset, a fraction in the interval (0,1)
/// ### Returns 
/// a tuple of four elements with the following structure:
///     1 and 2 are train and test tuples, respectively and are made of four elements, described as follow:
///     a. A vector of strings representing peptide sequence 
///     b. A vector of strings representing allele names 
///     c. A vector of strings representing tissue names            
///     d. A vector of integers (u8) representing the labels of each peptide, 1 represent binder and 0 represent non-binders 
///     3rd element:
///         is a hashmap of allele names named to its sequences 
///     4th element:
///         is the annotation table, see AnnotationTable for more details. 
pub fn preparation_prelude(input2prepare:&(Vec<String>,Vec<String>,Vec<String>),
        proteome:&HashMap<String,String>, path2cashed_db:String, 
        path2pseudo_seq:String, threshold:f32, fold_neg:u32, test_size:f32)->(
    (Vec<String>/* peptide */,Vec<String>/* allele name */,Vec<String>/* tissue name */,Vec<u8> /* label*/),
    (Vec<String>/* peptide */,Vec<String>/* allele name */,Vec<String>/* tissue name */,Vec<u8> /* label*/), 
    HashMap<String,String> /*Pseudo sequences map*/,
    HashMap<String, HashMap<String, ProteinInfo>>/* annotation table */
)
{

    // Load the cashed database 
    let database=read_cashed_db(&Path::new(&path2cashed_db)); 
    // load the pseudo_sequences database
    let pseudo_seq_map = read_pseudo_seq(&Path::new(&path2pseudo_seq));

    // build a target proteomes to analyze the data 
    let target_proteomes_per_tissue=create_negative_database_from_positive(&proteome,
        &input2prepare,&database,threshold);
    
    // group the data by alleles & tissue
    //-----------------------------------
    let positives_grouped_by_allele=group_by_allele_and_tissue(&input2prepare); 
    
    // prepare the test and the training data 
    //---------------------------------------

    let (train_data, test_data)=sample_negatives_from_positive_data_structure(
        positives_grouped_by_allele, &target_proteomes_per_tissue,fold_neg,test_size,&proteome); 

    (train_data,test_data,pseudo_seq_map,database)   
}

/// ### Summary 
/// Takes as an input a tuple of three vectors representing, the peptide sequence, the allele name and finally the target tissue 
/// and grouped them into a nested hashmap that allow easy and fast query by other functions. 
/// 
/// ### Parameters 
/// input2prepare: A tuple of three vectors representing, the peptide sequence, the allele name and finally the target tissue 
/// 
/// ### Return
/// group_by_alleles: A nested hashmap with the following structure:
///                                                     Allele name:
///                                                               |__ Tissue name (Key) --> peptides <Vec<String>> A vector of peptides sequences (Values). 
/// 
#[inline(always)]
fn group_by_allele_and_tissue(input2prepare:&(Vec<String>,Vec<String>,Vec<String>),)->HashMap<String,HashMap<String,Vec<String>>>
{
    // Check that the input is valid 
    if input2prepare.0.len()!=input2prepare.1.len() || input2prepare.0.len()!=input2prepare.2.len()
    {
        panic!("Critical error, the provided three tuples are of different length {},{},{} however, they all must have the same length.",input2prepare.0.len(),
        input2prepare.1.len(),input2prepare.2.len())
    }
    let unique_alleles=input2prepare.1.iter().collect::<HashSet<_>>(); 
    unique_alleles
        .into_par_iter()
        .map(|allele_name|
            {
                // Index target alleles  
                let mut target_rows=Vec::with_capacity(input2prepare.0.len());
                for idx in 0..input2prepare.0.len()
                {
                    if input2prepare.1[idx]==*allele_name{target_rows.push(idx)}
                }
                // Build a vectors of view over the table
                let mut view_on_table= Vec::with_capacity(target_rows.len()); 
                for idx in target_rows.iter()
                {
                    view_on_table.push((
                        &input2prepare.0[*idx],
                        &input2prepare.1[*idx],
                        &input2prepare.2[*idx]
                    ))
                }
                // compute unique tissues
                let unique_tissues=view_on_table.clone()
                                        .into_iter()
                                        .map(|(_,_,tissue)|tissue.clone().to_owned())
                                        .collect::<HashSet<_>>();
                let peptides_per_tissue=unique_tissues
                .iter()
                .map(|tissue_name|
                {
                    let peptides_from_tissue=view_on_table.clone()
                        .into_iter()
                        .filter(|(_,_,t_name)|t_name==&tissue_name)
                        .map(|(peptide,_,_)|peptide.clone().to_owned())
                        .collect::<Vec<String>>();
                    (tissue_name.clone(),peptides_from_tissue)
                })
                .collect::<HashMap<_,_>>(); 
                (allele_name.clone(),peptides_per_tissue)
            })
        .collect::<HashMap<_,_>>()
}

/// ### Summary
/// The function *ENCODES* the sub-cellular location into a multi-hot vector with shape of (1049,) and type of u8, 
/// each element represents whether the protein is observed into this location of not, 1 if it observed and zero otherwise 
/// ### Parameters
/// go_terms: a string representing all concatenated located_in GO term associated with the input protein  
/// ### Returns
/// a vector of shape (1049,) and type of u8, 
/// each element represents whether the protein is observed into this location of not, 1 if it observed and zero otherwise 
#[inline(always)]
pub fn get_sub_cellular_location(go_terms:&String)->Vec<u8>
{
    let terms=go_terms.split(";")
            .map(|elem| UNIQUE_GO_TERMS.iter().position(|unique_term|&elem==unique_term).unwrap_or(1048))
            .collect::<Vec<_>>(); 
    let mut results_vec=vec![0;1049];
    for term in terms
    {
        results_vec[term]=1
    }
    results_vec
}

/// ### Summary
/// Takes an annotation table as an input and return a hash-map containing tissue names as keys and context vectors as values
/// ### Parameters 
/// The annotation table, see annotation map for more details 
/// ### Return 
/// A hashmap representing the context vector of each gene in the cell, tissue name is represented as strings while the expression values is encoded as values
/// 
pub fn create_context_map(annoTable:&HashMap<String, HashMap<String, ProteinInfo>>)->HashMap<String,Vec<f32>>
{ 
    annoTable
        .par_iter()
        .map(|(tissue_name,tissue_info)|
        {
            // Create a vector of tuples linking transcript name with transcript expression value
            let mut txp_exp=tissue_info
                        .iter()
                        .map(|(txp_name,protein_info)|(txp_name.clone(),protein_info.get_expression()))
                        .collect::<Vec<(String,f32)>>();
            txp_exp.sort_by(|(_,exp_1),(_,exp_2)|exp_1.partial_cmp(exp_2).unwrap_or(Ordering::Equal));
            
            // Create the context vector  
            let context_vector=txp_exp.into_iter().map(|(_,exp)|exp).collect::<Vec<f32>>(); 

            (tissue_name.to_string(),context_vector)
        })
        .collect::<HashMap<_,_>>()
}

/// ### Summary
/// A helper function used for computing the distance to the nearest glycosylation site  
/// ### Parameter 
///     peptide_seq: The peptide sequence where the distance to glycosylation is to be computed 
///     protein_seq: The sequence of the parent protein 
///     glyco_site: a string representing the all glycosylation sites concatenated by a semi-colon, e.g. ; 
/// ### Return
/// distance to glycosylation: the distance to the nearest glycosylation sites. 
#[inline(always)]
pub fn compute_nearest_distance_to_glycosylation(peptide_seq:&String, protein_seq:&String, glyco_site:&String)->u32
{
    // First let's handle the case where the protein do not have a glycosylation sites, this will be encoded as a number equal to the protein length 
    // in this case, we return protein sequence and end executions 
    let glyco_sites=glyco_site.split(";").map(|elem|elem.parse::<i32>().unwrap()).collect::<Vec<_>>();
    if glyco_sites.len()==1 && glyco_sites[0] as usize == protein_seq.len()
    {
        return glyco_sites[0] as u32;
    }
    // otherwise we will have to compute the nearest glycosylation site. 
    // compute the distance to glycosylation 
    //--------------------------------------
    let positions = protein_seq.match_indices(peptide_seq).map(|(index,_)|index as i32).collect::<Vec<_>>();
    // compute the distance between all possible pair, i.e. all the locations where the peptide is allocated and all glycosylation site are known
    //-----------------------------------------------------------------------------------------
    positions
        .iter()
        .map(|position|
            {
                glyco_sites
                .iter()
                .map(|site|
                    {
                        match (*position..*position+peptide_seq.len() as i32).contains(site)
                        {
                            true=>0,
                            false=>(*site - *position).abs() as u32,
                        }
                    })
                .collect::<Vec<_>>()
            })
            .flatten()
            .collect::<HashSet<_>>()
            .into_iter()
            .min().unwrap()
}

/// ### Summary
/// Parse the input allele name and return a copy that is compatible with the pseudo-sequences table 
/// ### Parameters 
/// name: A string representing the input allele name
/// ### Return 
/// corrected allele names 
#[inline(always)]
pub fn parse_allele_names(name:String)->String
{
    match name.contains("DRB")
    {
        true =>
        {
            name.split("").collect::<Vec<_>>()[1].replace("", "").replace("*", "_").to_string()
        },
        false =>
        {
            "HLA-".to_owned() + &name.replace("*","").replace(":","")
        }
    }
}


/// ### Summary 
/// A helper function for computing and reading the quantitative data 
/// ------------------------------------------------------------------
/// ### Parameter
/// # path2file: The path to the reading files 
/// ### Returns
/// a tuple of three vectors, the first is the allele names, the second is peptides sequences and the third one if
/// a vector of floats representing the allele sequence
pub fn read_Q_table(path2file:&Path)->(Vec<String>,Vec<String>,Vec<f32>)
{
    // load the files to conduct the analysis
    //---------------------------------------
    let mut reader=csv::ReaderBuilder::new().delimiter(b'\t').from_path(path2file).unwrap(); // Read the files 

    // Allocate the vectors to fill the results
    //-----------------------------------------
    let mut allele_names=Vec::with_capacity(131_009); 
    let mut peptide_seq=Vec::with_capacity(131_009);
    let mut affinity=Vec::with_capacity(131_009);
    
    // Fill the results 
    //-----------------
    for record in reader.records()
    {
        let row=record.unwrap(); // getting a string record from the data
        allele_names.push(parse_allele_names(row[0].to_string())); 
        peptide_seq.push(row[1].to_string());
        affinity.push(row[2].parse::<f32>().unwrap())
    }
    (allele_names,peptide_seq,affinity)
}

/// ### Summary 
/// A convenient function for generating train and test dataset from the list of input positive examples.
/// ### Parameters:
/// positive_examples: List of string representing positive peptides 
/// test_size: The size of the test dataset, a float in the range (0,1). 
/// ### Returns 
/// A tuple of two vectors of strings, the first is the vector of train example and the second is the vector of test examples. 
pub fn split_positive_examples_into_test_and_train(positive_examples:&Vec<String>,test_size:f32)->(Vec<String>,Vec<String>)
{
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

    (train_seq,filtered_test_seq)
} 