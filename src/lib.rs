/// A collection of function used for parsing and preparing the input
/// 
/// 
/// 
/// 
use std::{collections::HashMap, path::Path};
use rayon::prelude::*;
use crate::functions::*; 
use ndarray::{Dim, Array};
use pyo3::prelude::*;
use crate::peptides::*; 
use numpy::{ToPyArray, PyArray};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

pub mod annotation_reader; 
pub mod protein_info; 
pub mod functions; 
pub mod peptides;
pub mod expression_db; 
pub mod train_seq_only; 


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
    Ok(annotate_all_proteins_in_one_tissues_using_cashed_db(&Path::new(&path2cashed_db),&target_protein,tissue_name).unwrap())
}


#[pyfunction]
fn get_annotated_protein_arrays<'py>(py:Python<'py>,
        path2cashed_db:String, target_protein:Vec<String>,
        tissue_name:String)->(Vec<String>,
            &'py PyArray<u8,Dim<[usize;2]>>,
            &'py PyArray<u16,Dim<[usize;2]>>,
            &'py PyArray<f32,Dim<[usize;2]>>
     )
{
    // build the annotations 
    //-----------------------
    let vec_annotation=annotate_all_proteins_in_one_tissues_using_cashed_db( &Path::new(&path2cashed_db),&target_protein,tissue_name).unwrap(); 
    // allocate the arrays
    let mut protein_ids=Vec::with_capacity(vec_annotation.len());
    let mut sub_cell_location=Vec::with_capacity(vec_annotation.len()); 
    let mut distance2glyco=Vec::with_capacity(vec_annotation.len()); 
    let mut gene_exp=Vec::with_capacity(vec_annotation.len()); 
    // fill the vectors 
    //-----------------
    for (protein_id,subcell_loc,d2g,exp) in vec_annotation
    {
        protein_ids.push(protein_id);
        sub_cell_location.push(subcell_loc);
        distance2glyco.push(distance2glyco); 
        gene_exp.push(exp); 
    }
    // calling the encoders
    let encoded_subcell_loc=encode_subcell_locations();
    let encoded_d2g=encode_distance_to_glyco(); 
    let gene_exp=encode_gene_expression(); 
}


/*
#[pyfunction]
fn prepare_training_data(){}*/





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
    Ok(group_peptides_by_parent_rs(peptides,proteome))
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
    Ok(group_by_9mers_rs(peptides))
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



/* 
#[pyfunction]
fn generate_a_train_arrays_by_proteome_sampling<'py>(py:Python<'py>,
            positive_examples:Vec<String>,fold_neg:u32, max_len:usize)->(&'py PyArray<u8,Dim<[usize;2]>>,&'py PyArray<u8,Dim<[usize;2]>>)
{

}*/

/// encode_sequence(seq:List[str],max_len:int)->np.ndarray
/// --
/// A multi-threaded Rust sided function that used for encoding input sequences into 2D tensors with shape of (num_seq,seq_len)
/// 
/// 
#[pyfunction]
fn encode_sequence<'py>(py:Python<'py>,seq:Vec<String>,max_len:usize)->&'py PyArray<u8,Dim<[usize;2]>>
{
    encode_sequence_rs(seq,max_len).to_pyarray(py)
}

/// A Python module implemented in Rust.
#[pymodule]
fn RustDB(_py: Python, m: &PyModule) -> PyResult<()> 
{
    m.add_function(wrap_pyfunction!(annotate_proteins, m)?)?;
    m.add_function(wrap_pyfunction!(cash_database,m)?)?;
    m.add_function(wrap_pyfunction!(annotate_proteins_using_cashed_db,m)?)?;
    m.add_function(wrap_pyfunction!(group_peptide_by_protein,m)?)?;
    m.add_function(wrap_pyfunction!(group_by_9mers,m)?)?; 
    m.add_function(wrap_pyfunction!(generate_a_train_db_by_shuffling,m)?)?;
    m.add_function(wrap_pyfunction!(encode_sequence,m)?)?;
    m.add_function(wrap_pyfunction!(generate_a_train_arrays_by_shuffling,m)?)?;
    Ok(())
}