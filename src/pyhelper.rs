/// Load the modules
///-----------------
use std::{path::Path, collections::HashMap};
use ndarray::Dim;
use numpy::{PyArray, ToPyArray};
use pyo3::{pyfunction, PyResult, Python};
use crate::{peptides::{generate_a_train_db_by_shuffling_rs, group_peptides_by_parent_rs, group_by_9mers_rs, encode_sequence_rs}, 
            functions::{annotate_all_proteins_in_one_tissue, cash_database_to_disk, read_cashed_db}, utils::{create_negative_database_from_positive, group_by_allele_and_tissue, sample_negatives_from_positive_data_structure_no_test_split}
        };

//-----------------------------------------|
// declaring and implementing the functions| 
//-----------------------------------------|

/// ### Signature
/// annotate_proteins(path:str,target_protein:List[str],tissue_name:str)->List[Tuple[str,str,str,float32]]
/// ### Summary
/// A wrapper function used to automate the task of parsing the annotation table and generating the annotations for a collection of proteins 
/// in a specific tissue.
/// ### Executioner 
/// This function is a wrapper function for the Rust function annotate_all_proteins_in_one_tissue, for more details regard the execution logic check 
/// the documentation for this function.
/// ### Parameters
/// path: The path to load the annotation table
/// target_protein: A list of target proteins to annotate
/// tissue_name: The name of the tissue where annotation information should be extracted
/// ### Returns
/// A list of tuples each of which is composite of 4 elements:
///     1. A string representing the protein name, i.e. uniprot ID 
///     1. A String representing the subcellular location
///     2. A String representing the distance to glycosylation 
///     3. A float representing the gene expression 
/// --
#[pyfunction]
fn annotate_proteins(path:String,target_proteins:Vec<String>,
    tissue_name:String)->PyResult<Vec<(String,String,String,f32)>>
{
    Ok(annotate_all_proteins_in_one_tissue(&Path::new(&path),&target_proteins,tissue_name))
    
}

/// ### Signature
///  cash_database(path2db:str, path2res:str)-> None
/// ### Summary
/// takes the path to an annotation table, parse and create an hashmap out of it finally it write the hashmap to the disk using serde.
///  This enable the database to be reloaded relatively fast for subsequent usages.
/// ### Executioner 
/// This function is a wrapper function for the Rust function cash_database_to_disk, for more details regard the execution logic check 
/// the documentation for this function.
/// ### Parameters
/// path2db: The path to load the database, i.e. the annotation table 
/// path2res: The path to write the results, i.e. the cashed database
/// --
#[pyfunction]
fn cash_database(path2db:String,path2res:String)->PyResult<()>
{
    Ok(cash_database_to_disk(&Path::new(&path2db), &Path::new(&path2res)))
}
/// ### Signature
/// annotate_proteins_using_cashed_db(path:str,target_protein:List[str],tissue_name:str)->List[Tuple[str,str,str,float32]]
/// ### Summary
/// A wrapper function used to automate the task of parsing the annotation table and generating the annotations for a collection of proteins 
/// in a specific tissue.
/// ### Executioner 
/// This function is a wrapper function for the Rust function annotate_all_proteins_in_one_tissues_using_cashed_db, for more details regard the execution logic check 
/// the documentation for this function.
/// ### Parameters
/// path: The path to load the *CASHED* annotation table
/// target_protein: A list of target proteins to annotate
/// tissue_name: The name of the tissue where annotation information should be extracted
/// ### Returns
/// A list of tuples each of which is composite of 4 elements:
///     1. A string representing the protein name, i.e. uniprot ID 
///     1. A String representing the subcellular location
///     2. A String representing the distance to glycosylation 
///     3. A float representing the gene expression 
/// --
#[pyfunction]
fn annotate_proteins_using_cashed_db(path2cashed_db:String, target_protein:Vec<String>,
    tissue_name:String)->PyResult<Vec<(String,String,String,f32)>>
{
    Ok(annotate_all_proteins_in_one_tissue(&Path::new(&path2cashed_db),&target_protein,tissue_name))
}



/// ### Signature
/// group_peptide_by_protein(peptide:List[str],proteome:Dict[str,str])->Dict[str,List[str]]
/// ### Summary
/// The function takes a list of peptides as an input and a dict of protein names mapped to protein sequences as values and return a dict containing
/// the peptides as keys and list of protein ids where the proteins were observed as values.
/// ### Executioner 
/// This function is a wrapper function for the Rust function group_peptide_by_protein, for more details regard the execution logic check 
/// the documentation for this function.
/// ### Parameters
/// peptides: A list of peptide sequences
/// proteome: A dict of protein-name as keys and proteins-sequences as values 
/// ### Returns
/// a dict containing the peptides as keys and list of protein ids where the proteins were observed as values.   
///---- 
#[pyfunction]
fn group_peptide_by_protein(peptides:Vec<String>,proteome:HashMap<String,String>)->PyResult<HashMap<String,Vec<String>>>
{
    Ok(group_peptides_by_parent_rs(&peptides,&proteome))
}
/// ### Signature
/// group_by_9mers(peptide:List[str])->Dict[str,List[str]]
/// ### Summary
/// group input peptide by the 9 mers cores, the function first create a set of all unique 9 mers in the input list of peptides afterword
/// it loops over all peptides and create a dict with 9mers are keys and a collection of peptide sequences where the 9 mers was observed as a value.
/// ### Executioner 
/// This function is a wrapper function for the Rust function group_by_9mers_rs, for more details regard the execution logic check 
/// the documentation for this function.
/// ### Parameters
/// peptides: A list of peptide sequences
/// ### Returns
/// a dict containing the peptides as keys and and a list of peptide sequences where the 9 mers was observed as a value.  
/// 
#[pyfunction]
fn group_by_9mers(peptides:Vec<String>)->PyResult<HashMap<String,Vec<String>>>
{
    Ok(group_by_9mers_rs(&peptides))
}

/// ### Signature
/// generate_a_train_db_by_shuffling(positive_examples:List[str], fold_neg:int)->Tuple[List[str],List[str]]
/// ### Summary
/// create a train database from a collection of positive examples, where the function first reads an input list of positive examples
/// create a matched database of negative examples through shuffling and return a tuple of two lists, the first contain peptide sequences 
/// while the second contain numerical labels.  
/// ### Executioner 
/// This function is a wrapper function for the Rust function generate_a_train_db_by_shuffling_rs, for more details regard the execution logic check 
/// the documentation for this function.
/// ### Parameters
/// positive_examples: A list of peptide sequences representing positive examples
/// fold_neg: The ration of the negative, i.e. shuffled generated, to the positive examples. 
/// ### Returns
/// a tuple of two lists, the first contain peptide sequences while the second contain numerical labels.  
#[pyfunction]
fn generate_a_train_db_by_shuffling(positive_examples:Vec<String>, fold_neg:u32)->PyResult<(Vec<String>,Vec<u8>)>
{
    Ok(generate_a_train_db_by_shuffling_rs(positive_examples,fold_neg))
}

/// ### Signature
/// encode_sequence(seq:List[str],max_len:int)->np.ndarray
/// ### Summary
/// Encodes and pads a list of peptides into an array of shape (num_peptides,max_len) and type u8.
/// The encoding scheme is hardcoded into the code base. to change please edit the functions in peptide and recompile the code. 
/// ### Parameters
/// positive_examples: A list of peptide sequences that will be numerically encoded
/// max_len: The maximum length of each peptide, shorter sequences are zero padded and longer sequences are trimmed. 
/// ### Returns
///  A NumPy array of shape (num_peptides,max_len) and type np.uint8
#[pyfunction]
fn encode_sequence<'py>(py:Python<'py>,seq:Vec<String>,max_len:usize)->&'py PyArray<u8,Dim<[usize;2]>>
{
    encode_sequence_rs(seq,max_len).to_pyarray(py)
}

/// ### Signature
/// sample_negatives_from_positives_no_test(input: Tuple[
///                                                     List[str], # Input list of Peptides
///                                                     List[str], # Input list of Allele names
///                                                     List[str], # Input list of Tissue names 
///                                                     ],
///                                         proteome: Dict[str,str], # A Lookup table of protein to peptide names,
///                                         path2cashed_bd: str, # The path to a cashed database
///                                         fold_neg: int, # The ratio of negative to positives,
///                                         threshold:f32, # the minimum expression level needed, for generating the negatives              
///                                                 )->Tuple[
///                                                 List[str], # generated peptides
///                                                 List[str], # allele names,
///                                                 List[str], # tissue names,
///                                                 List[str], # label for each peptide, {1:binder, 0:non_binder}
///                                             ])
/// ### Summary
/// The function samples negative and return the original and the sampled negative as a one tuple, see signature above for more details. 
/// ### ---------
#[pyfunction]
fn sample_negatives_from_positives_no_test(input:(Vec<String>,Vec<String>,Vec<String>), 
    proteome:HashMap<String,String>, path2cashed_db:String, fold_neg:u32, threshold:f32)->(Vec<String>/* peptide */, 
        Vec<String>/* allele name */, Vec<String>/* tissue name */, Vec<u8> /* label*/)
{
    // Load the cashed database 
    let database=read_cashed_db(&Path::new(&path2cashed_db)); 
    // load the pseudo_sequences database
    // build a target proteomes to analyze the data 
    let target_proteomes_per_tissue=create_negative_database_from_positive(&proteome,
        &input,&database,threshold);
    // group the data by alleles & tissue
    //-----------------------------------
    let positives_grouped_by_allele=group_by_allele_and_tissue(&input); 
    // Return the results
    //-------------------
    sample_negatives_from_positive_data_structure_no_test_split(positives_grouped_by_allele, &target_proteomes_per_tissue, fold_neg,&proteome)
}