/// *** Library Prelude ***
/// ---------------------- 
/// Brief: A library for preparing data sequences and Multi-omics data for training peptide HLA-II interactions models 
/// Author: Hesham ElAbd 
/// Contact: h.elabd@ikmb.uni-kiel.de
/// Copyrights: Institute of clinical molecular biology, Kiel, Germany.
/// Version: 0.1.0 pre-alpha
/// Release data: 10.02.2022
/// Initial Release date: 10.02.2022
/// Rust Code: The functions in this module provide a thin wrapper a round the rust code which is used for executing all the heavy-lifting jobs.
///         To skip the current module and work with the Rust-code directly or to extend the rust code, check the code defined in the file omics_builder.rs.
/// Bug Reporting and tracking: 
///         incase of any bug, contact the developer using the contact information defined above or open an issue at the github page (https://github.com/ikmb/O-Link-).
/// Change Track:
///     No update to the 0.1.0 pre-alpha version has been documented.
/// NOTE:
///     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
///     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
///     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
///     
///     CHECK THE LICENSE FOR MORE DETAILS  
/// Details:
///     To be written 
///-----------------------------------------------

/// Exported functions
/// ------------------
/// The xxx library provides an interface to prepare and encode the datasets for training peptide HLA-II interaction and for running predictions on  
/// the training datasets. The following functions are exported: 
///     1.  generate_train_ds_shuffling_sm: which generate a training dataset made of negatives through shuffling of the positive peptides then
///        numerically encodes it train and test dataset and then return two arrays, one representing the encoded sequences and representing the labels. 
///         Further information can be obtained by calling the help function of the module.
///     2.  
///      
///      
/// 
/// 
/// 
/// 
/// 
///   
// Import the binder functions 
//----------------------------
use pyo3::prelude::*;

// Export the modules of the library
//----------------------------------
pub mod annotation_reader; 
pub mod protein_info; 
pub mod functions; 
pub mod peptides;
pub mod train_seq_only; 
pub mod geneExpressionIO;
pub mod train_mo_model; 
pub mod utils;
pub mod constants;
pub mod omics_builder; 
pub mod sequence_builder; 
pub mod pyhelper;

// Import the python wrappers
use crate::train_mo_model::*;
use crate::train_seq_only::*; 
use crate::pyhelper::*; 

// add functions to the Py Module

#[pymodule]
fn OLink(_py: Python, m: &PyModule) -> PyResult<()> 
{
    m.add_function(wrap_pyfunction!(generate_train_ds_shuffling_sm, m)?)?;
    m.add_function(wrap_pyfunction!(generate_a_train_arrays_by_shuffling, m)?)?;
    m.add_function(wrap_pyfunction!(generate_train_ds_proteome_sampling_sm, m)?)?;
    m.add_function(wrap_pyfunction!(generate_train_ds_same_protein_sampling_sm, m)?)?;
    m.add_function(wrap_pyfunction!(generate_train_ds_expressed_protein_sampling_sm, m)?)?;
    m.add_function(wrap_pyfunction!(generate_train_ds_pm, m)?)?;
    m.add_function(wrap_pyfunction!(generate_train_ds_Qd, m)?)?;
    m.add_function(wrap_pyfunction!(generate_train_based_on_seq_exp, m)?)?;
    m.add_function(wrap_pyfunction!(generate_train_based_on_seq_exp_subcell, m)?)?;
    m.add_function(wrap_pyfunction!(generate_train_based_on_seq_exp_subcell_context, m)?)?;
    m.add_function(wrap_pyfunction!(generate_train_based_on_seq_exp_subcell_loc_context_d2g, m)?)?;
    m.add_function(wrap_pyfunction!(annotate_proteins, m)?)?;
    m.add_function(wrap_pyfunction!(cash_database, m)?)?;
    m.add_function(wrap_pyfunction!(annotate_proteins_using_cashed_db, m)?)?;
    m.add_function(wrap_pyfunction!(group_peptide_by_protein, m)?)?;
    m.add_function(wrap_pyfunction!(group_by_9mers, m)?)?;
    m.add_function(wrap_pyfunction!(encode_sequence, m)?)?;

    Ok(())
}