/// The inference engine function of the OmLiT library
use std::{path::Path, collections::HashMap};

use chrono::Utc;
use ndarray::Dim;
use numpy::{PyArray, ToPyArray};
use pyo3::prelude::*;

use crate::{peptides::encode_sequence_rs, functions::read_cashed_db, utils::{read_pseudo_seq}, omics_builder::prepare_data_for_seq_exp_subcellular_context_d2g_data}; 

/// ### Signature 
/// Encode_sequences(input_seq:List[str],max_len:usize)->np.ndarray
/// ### Summary 
/// Numerically encode an input collection of sequences into arrays of integers with shape num_examples, by max length. 
/// ### Parameters
/// input_peptides: an input collection of peptides sequences
/// max_len: the maximum length of the input peptide, longer peptides are trimmed while shorter are pre-padded with zeros. 
/// ### Returns 
/// Returns a 2D NumPy array with shape of (num_training_examples,max_length) and a type np.uint8 
#[pyfunction]
pub fn encode_sequences<'py>(py:Python<'py>, 
            input_peptides:Vec<String>,max_len:usize)->&'py PyArray<u8,Dim<[usize;2]>>
{
    encode_sequence_rs(input_peptides,max_len).to_pyarray(py)
}

/// ### Signature 
/// annotate_and_encode_input_sequences(inputs:Tuple[List[str] # Peptide name 
///                                     ,List[str] # Allele name 
///                                     ,List[str] # Tissue name
///                                     ,List[int] # Labels    
///                                     ],
///                                     max_len:int # maximum peptide length, shorter peptides are padded to this length while longer are trimmed,
///                                     proteome: Dict[str,str] # a hashmap of protein ids to protein sequence,
///                                     path2cashed_db: str, the path to load the cashed database 
///                                     path2pseudo_seq: str, the path to load the pseudo sequences, used for generating the any array of pseudo sequences
///                                     )->Tuple[
///                                         Tuple[
///                                             np.ndarray[None,max_len] # encoded peptide sequences,
///                                             np.ndarray[None,34] # encoded protein sequences,
///                                             np.ndarray[None,1049]# encoded subcellular locations,
///                                             np.ndarray[None,16_519] # the context vector,
///                                             np.ndarray[None,1] # the distance to the nearest glycosylation site
///                                             np.ndarray[None,1] # labels for each data point 
///                                         ], # this tuple represent the mapped and correctly encoded protein sequences
///                                         Tuple[
///                                               List[str], # List of peptides
///                                               List[str], # List of alleles
///                                               List[str], # List of Tissue names
///                                               List[str]  # List of labels
///                                               ]
///                                     ]
/// ### Summary 
/// An inference preprocessing engine that can be used to annotate an input collection of peptides and returns two tuple, the first tuple represent the annotated and encoded input peptides,
/// meanwhile the second tuple represent the unmapped input data points.   
#[pyfunction]
pub fn annotate_and_encode_input_sequences<'py>(py:Python<'py>, 
    input:(Vec<String>/* List of peptides*/,Vec<String> /*List of alleles*/,Vec<String> /*list of tissue names*/,Vec<u8>/*Labels*/),
    max_len:usize,
    proteome:HashMap<String,String>,
    path2cashed_db:String,
    path2pseudo_seq:String,
)->(
        (&'py PyArray<u8,Dim<[usize;2]>> /* Peptide sequences*/, &'py PyArray<u8,Dim<[usize;2]>> /*Pseudo-sequences*/,
         &'py PyArray<f32,Dim<[usize;2]>> /* Expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Sub-cellular location values*/,
         &'py PyArray<f32,Dim<[usize;2]>> /* Context vectors*/, &'py PyArray<u32,Dim<[usize;2]>> /*Distance to glycosylation*/,
         &'py PyArray<u8,Dim<[usize;2]>> /* Labels */),
        (Vec<String>/* Peptides*/, Vec<String>/*Alleles*/,Vec<String>/*Tissue names*/, Vec<u8>/*Labels*/)// unmapped train data points 
    )
{
    // Load the cashed database
    println!("Loading the annotation and Pseudo Sequence database ... {}",Utc::now());  
    let database=read_cashed_db(&Path::new(&path2cashed_db));
    let pseudo_seq_map = read_pseudo_seq(&Path::new(&path2pseudo_seq));
    
    // Build a target proteomes to analyze the data 
    println!("Annotating {} data points starting at ... {}",input.0.len(),Utc::now()); 
    let (encoded_results, unmapped_results)=prepare_data_for_seq_exp_subcellular_context_d2g_data(input,
        &pseudo_seq_map,max_len,&proteome,&database); 
    
    // preparing input arrays ...
    //---------------------------
    (
        /* Generate the mapped arrays*/
        (encoded_results.0.to_pyarray(py),encoded_results.1.to_pyarray(py),
        encoded_results.2.to_pyarray(py),encoded_results.3.to_pyarray(py),
        encoded_results.4.to_pyarray(py),encoded_results.5.to_pyarray(py),
        encoded_results.6.to_pyarray(py)),
        /* Generate the unmapped arrays*/
        unmapped_results
    ) 
}