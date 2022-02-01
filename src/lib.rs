/// A collection of function used for parsing and preparing the input
use std::{collections::HashMap, path::Path};
use crate::functions::*; 
use ndarray::Dim;
use pyo3::prelude::*;
use crate::peptides::*; 
use numpy::{ToPyArray, PyArray};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
pub mod annotation_reader; 
pub mod protein_info; 
pub mod functions; 
pub mod peptides;
pub mod expression_db; 



#[pyfunction]
fn annotate_proteins(path:String,target_protein:Vec<String>,
    tissue_name:String)->PyResult<Vec<(String,String,String,f32)>>
{
    Ok(annotate_all_proteins_in_one_tissue(&Path::new(&path),&target_protein,tissue_name).unwrap())
    
}

#[pyfunction]
fn cash_database(path2db:String,path2res:String)->PyResult<()>
{
    Ok(cash_database_to_disk(&Path::new(&path2db), &Path::new(&path2res)))
}

#[pyfunction]
fn annotate_proteins_using_cashed_db(path2cashed_db:String, target_protein:Vec<String>,
    tissue_name:String)->PyResult<Vec<(String,String,String,f32)>>
{
    Ok(annotate_all_proteins_in_one_tissues_using_cashed_db(&Path::new(&path2cashed_db),&target_protein,tissue_name).unwrap())
    
}

#[pyfunction]
fn group_peptide_by_protein(peptides:Vec<String>,proteome:HashMap<String,String>)->PyResult<HashMap<String,Vec<String>>>
{
    Ok(group_peptides_by_parent_rs(peptides,proteome))
}

#[pyfunction]
fn group_by_9mers(peptides:Vec<String>)->PyResult<HashMap<String,Vec<String>>>
{
    Ok(group_by_9mers_rs(peptides))
}


#[pyfunction]
fn generate_a_train_db_by_shuffling(positive_examples:Vec<String>, fold_neg:u32)->PyResult<(Vec<String>,Vec<u8>)>
{
    Ok(generate_a_train_db_by_shuffling_rs(positive_examples,fold_neg))
}

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
fn RustDB(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(annotate_proteins, m)?)?;
    m.add_function(wrap_pyfunction!(cash_database,m)?)?;
    m.add_function(wrap_pyfunction!(annotate_proteins_using_cashed_db,m)?)?;
    m.add_function(wrap_pyfunction!(group_peptide_by_protein,m)?)?;
    m.add_function(wrap_pyfunction!(group_by_9mers,m)?)?; 
    m.add_function(wrap_pyfunction!(generate_a_train_db_by_shuffling,m)?)?;
    m.add_function(wrap_pyfunction!(encode_sequence,m)?)?;
    Ok(())
}