use std::collections::HashMap;
use ndarray::Dim;
use numpy::{PyArray, ToPyArray};
use pyo3::Python;
use crate::{utils::*, omics_builder::*};

pub fn generate_train_based_on_seq_exp<'py>(py:Python<'py>,
        input2prepare:(Vec<String>,Vec<String>,Vec<String>),
                    proteome:HashMap<String,String>, path2cashed_db:String, 
                    path2pseudo_seq:String, max_len:usize,
        threshold:f32, fold_neg:u32, test_size:f32
    )->(
        /* Train Tensors */
        (&'py PyArray<u8,Dim<[usize;2]>> /*Train pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train encoded sequence */,
         &'py PyArray<f32,Dim<[usize;2]>> /*Train expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /* Train labels */), 
        
        /* Test Tensors */
        (&'py PyArray<u8,Dim<[usize;2]>> /*Test pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Test encoded sequence */,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /* Test labels */), 

        (Vec<String>,Vec<String>,Vec<String>, Vec<u8>),// unmapped train data points 
        (Vec<String>,Vec<String>,Vec<String>, Vec<u8>) // unmapped test data points
    )
{
    // Prepare the prelude for the encoders
    //------------------------------------- 
    let (train_data, test_data, pseudo_seq
            ,anno_table)=preparation_prelude(&input2prepare,&proteome,path2cashed_db,path2pseudo_seq,
            threshold,fold_neg,test_size);
    
    // encode and prepare the training data
    //-------------------------------------
    let (encoded_train_data,unmapped_train_data)=prepare_data_for_seq_and_expression_data(train_data,
        &pseudo_seq,max_len,&proteome,&anno_table); 
    
    let(encoded_test_data,unmapped_test_data)=prepare_data_for_seq_and_expression_data(test_data,
        &pseudo_seq,max_len,&proteome,&anno_table); 

    //return the results
    //------------------
    (
        (encoded_train_data.0.to_pyarray(py),encoded_train_data.1.to_pyarray(py),encoded_train_data.2.to_pyarray(py),encoded_train_data.3.to_pyarray(py)),
        (encoded_test_data.0.to_pyarray(py),encoded_test_data.1.to_pyarray(py),encoded_test_data.2.to_pyarray(py),encoded_test_data.3.to_pyarray(py)),
        unmapped_train_data,
        unmapped_test_data
    )
}

pub fn generate_train_based_on_seq_exp_subcell<'py>(py:Python<'py>,
input2prepare:(Vec<String>,Vec<String>,Vec<String>),
            proteome:HashMap<String,String>, path2cashed_db:String, 
            path2pseudo_seq:String, max_len:usize,
            threshold:f32, fold_neg:u32, test_size:f32
    )->(
        (&'py PyArray<u8,Dim<[usize;2]>> /*Train pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train encoded sequence */,
         &'py PyArray<f32,Dim<[usize;2]>> /*Train expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train sub-cellular location values*/,
         &'py PyArray<u8,Dim<[usize;2]>> /* Train labels */),
        
        (&'py PyArray<u8,Dim<[usize;2]>> /*Test pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Test encoded sequence */,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train sub-cellular location values*/,
        &'py PyArray<u8,Dim<[usize;2]>> /* Test labels */),

        (Vec<String>,Vec<String>,Vec<String>, Vec<u8>),// unmapped train data points 
        (Vec<String>,Vec<String>,Vec<String>, Vec<u8>) // unmapped test data points
    )
{
    // Prepare the prelude for the encoders
    //------------------------------------- 
    let (train_data, test_data, pseudo_seq
        ,anno_table)=preparation_prelude(&input2prepare,&proteome,path2cashed_db,path2pseudo_seq,
        threshold,fold_neg,test_size);
    
    // encode and prepare the training data
    //-------------------------------------
    let (encoded_train_data,unmapped_train_data)=prepare_data_for_seq_exp_and_subcellular_data(train_data,
        &pseudo_seq,max_len,&proteome,&anno_table); 
    
    let(encoded_test_data,unmapped_test_data)=prepare_data_for_seq_exp_and_subcellular_data(test_data,
        &pseudo_seq,max_len,&proteome,&anno_table); 

    //return the results
    //------------------
    (
        (encoded_train_data.0.to_pyarray(py),encoded_train_data.1.to_pyarray(py),encoded_train_data.2.to_pyarray(py),encoded_train_data.3.to_pyarray(py),encoded_train_data.4.to_pyarray(py)),
        (encoded_test_data.0.to_pyarray(py),encoded_test_data.1.to_pyarray(py),encoded_test_data.2.to_pyarray(py),encoded_test_data.3.to_pyarray(py),encoded_test_data.4.to_pyarray(py)),
        unmapped_train_data,
        unmapped_test_data
    )
}


pub fn generate_train_based_on_seq_exp_subcell_context<'py>(py:Python<'py>,
        input2prepare:(Vec<String>,Vec<String>,Vec<String>),
        proteome:HashMap<String,String>, path2cashed_db:String, 
        path2pseudo_seq:String, max_len:usize,threshold:f32,
        fold_neg:u32, test_size:f32
    )->(
        (&'py PyArray<u8,Dim<[usize;2]>> /*Train pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train encoded sequence */,
         &'py PyArray<f32,Dim<[usize;2]>> /*Train expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train sub-cellular location values*/,
         &'py PyArray<f32,Dim<[usize;2]>> /*Train context vectors*/,&'py PyArray<u8,Dim<[usize;2]>> /* Train labels */),
        
        (&'py PyArray<u8,Dim<[usize;2]>> /*Test pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Test encoded sequence */,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train sub-cellular location values*/,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test context vectors*/, &'py PyArray<u8,Dim<[usize;2]>> /* Test labels */),

        (Vec<String>,Vec<String>,Vec<String>, Vec<u8>),// unmapped train data points 
        (Vec<String>,Vec<String>,Vec<String>, Vec<u8>) // unmapped test data points
    )
{
    // Prepare the prelude for the encoders
    //------------------------------------- 
    let (train_data, test_data, pseudo_seq
        ,anno_table)=preparation_prelude(&input2prepare,&proteome,path2cashed_db,path2pseudo_seq,
        threshold,fold_neg,test_size);
    // create the annotations
    //-----------------------
    // encode and prepare the training data
    //-------------------------------------
    let (encoded_train_data,unmapped_train_data)=prepare_data_for_seq_exp_subcellular_and_context_data(train_data,
        &pseudo_seq,max_len,&proteome,&anno_table); 
    
    let(encoded_test_data,unmapped_test_data)=prepare_data_for_seq_exp_subcellular_and_context_data(test_data,
        &pseudo_seq,max_len,&proteome,&anno_table); 

    //return the results
    //------------------
    (
        (encoded_train_data.0.to_pyarray(py),encoded_train_data.1.to_pyarray(py),encoded_train_data.2.to_pyarray(py),encoded_train_data.3.to_pyarray(py),encoded_train_data.4.to_pyarray(py),encoded_train_data.5.to_pyarray(py)),
        (encoded_test_data.0.to_pyarray(py),encoded_test_data.1.to_pyarray(py),encoded_test_data.2.to_pyarray(py),encoded_test_data.3.to_pyarray(py),encoded_test_data.4.to_pyarray(py),encoded_test_data.5.to_pyarray(py)),
        unmapped_train_data,
        unmapped_test_data
    )

}


pub fn generate_train_based_on_seq_exp_subcell_loc_context_d2g<'py>(py:Python<'py>,
            input2prepare:(Vec<String>,Vec<String>,Vec<String>),
            proteome:HashMap<String,String>, path2cashed_db:String, 
            path2pseudo_seq:String, max_len:usize,
            threshold:f32, fold_neg:u32, test_size:f32
    )->(
        (&'py PyArray<u8,Dim<[usize;2]>> /*Train peptide sequences*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train pseudo-sequences*/,
         &'py PyArray<f32,Dim<[usize;2]>> /*Train expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train sub-cellular location values*/,
         &'py PyArray<f32,Dim<[usize;2]>> /*Train context vectors*/, &'py PyArray<u32,Dim<[usize;2]>> /*Train distance to glycosylation*/,
         &'py PyArray<u8,Dim<[usize;2]>> /* Train labels */),
        
        (&'py PyArray<u8,Dim<[usize;2]>> /*Test pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Test encoded sequence */,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Test sub-cellular location values*/,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test context vectors*/,&'py PyArray<u32,Dim<[usize;2]>> /*Test distance to glycosylation*/,
        &'py PyArray<u8,Dim<[usize;2]>> /* Test labels */), 

        (Vec<String>,Vec<String>,Vec<String>, Vec<u8>),// unmapped train data points 
        (Vec<String>,Vec<String>,Vec<String>, Vec<u8>) // unmapped test data points
    )
{
    // Prepare the prelude for the encoders
    //------------------------------------- 
    let (train_data, test_data, pseudo_seq
        ,anno_table)=preparation_prelude(&input2prepare,&proteome,path2cashed_db,path2pseudo_seq,
        threshold,fold_neg,test_size);
    
    // encode and prepare the training data
    //-------------------------------------
    let (encoded_train_data,unmapped_train_data)=prepare_data_for_seq_exp_subcellular_context_d2g_data(train_data,
        &pseudo_seq,max_len,&proteome,&anno_table); 
    
    let(encoded_test_data,unmapped_test_data)=prepare_data_for_seq_exp_subcellular_context_d2g_data(test_data,
        &pseudo_seq,max_len,&proteome,&anno_table); 
    
    //return the results
    //------------------
    (
        (encoded_train_data.0.to_pyarray(py),encoded_train_data.1.to_pyarray(py),encoded_train_data.2.to_pyarray(py),
        encoded_train_data.3.to_pyarray(py),encoded_train_data.4.to_pyarray(py),encoded_train_data.5.to_pyarray(py),encoded_train_data.6.to_pyarray(py)),
        (encoded_test_data.0.to_pyarray(py),encoded_test_data.1.to_pyarray(py),encoded_test_data.2.to_pyarray(py),
        encoded_test_data.3.to_pyarray(py),encoded_test_data.4.to_pyarray(py),encoded_test_data.5.to_pyarray(py),encoded_test_data.6.to_pyarray(py)),
        unmapped_train_data,
        unmapped_test_data
    )
}

