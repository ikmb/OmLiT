/// *** Library Prelude ***
/// ---------------------- 
/// Brief: A library for preparing data sequences and Multi-omics data for training peptide HLA-II interactions models 
/// Author: Hesham ElAbd 
/// Contact: h.elabd@ikmb.uni-kiel.de
/// Copyrights: Institute of clinical molecular biology, Kiel, Germany.
/// Version: 0.1.1 alpha
/// Release data: 04.03.2022
/// Initial Release date: 10.02.2022
/// Rust Code: The functions in this module provide a thin wrapper a round the rust code which is used for executing all the heavy-lifting jobs.
///         To skip the current module and work with the Rust-code directly or to extend the rust code, check the code defined in the file omics_builder.rs.
/// Bug Reporting and tracking: 
///         incase of any bug, contact the developer using the contact information defined above or open an issue at the github page (https://github.com/ikmb/O-Link-).
/// Change Track:
///     No update to the 0.1.0 pre-alpha version has been documented.
///     Version 0.1.1:
///         The following changes were included into the modules
///             a. Added an inference engine as defined in the (inference_engine) module of the Rust-code and added two python-sided functions to bind the inference engine to the python code. 
///             b. Increase the ability to customize the training of the data structure, were python-sided python functions were build to hookup into different steps of the preprocessing Rust pipeline
///             c. increasing the documentation of the underlining library structure.
///             
/// NOTE:
///     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
///     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
///     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
///     
///     CHECK THE LICENSE FOR MORE DETAILS  
///-----------------------------------------------------------------------------------------------------
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
pub mod inference_engine; 

// Import the python wrappers
use crate::train_mo_model::*;
use crate::train_seq_only::*; 
use crate::pyhelper::*; 
use crate::inference_engine::*; 

// add functions to the Py Module
#[pymodule]
fn OmLiT(_py: Python, m: &PyModule) -> PyResult<()> 
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
    m.add_function(wrap_pyfunction!(encode_sequences,m)?)?;
    m.add_function(wrap_pyfunction!(sample_negatives_from_positives_no_test,m)?)?; 
    m.add_function(wrap_pyfunction!(annotate_and_encode_input_sequences,m)?)?; 
    m.add_function(wrap_pyfunction!(annotate_and_encode_input_sequences_no_label,m)?)?; 
    Ok(())
}