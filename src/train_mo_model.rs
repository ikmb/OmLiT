use std::{collections::{HashMap, HashSet}, path::Path, hash::Hash, iter::FromIterator, ops::RangeBounds, alloc::LayoutErr};

use ndarray::{Dim, Array, Shape, Ix2, IndexLonger};
use numpy::{PyArray, ToPyArray};
use pyo3::Python;
use rand::prelude::{SliceRandom, IteratorRandom};
use rayon::prelude::*; 
use crate::constants::UNIQUE_GO_TERMS; 
use crate::{functions::read_cashed_db, protein_info::ProteinInfo, utils::read_pseudo_seq, peptides::{group_by_9mers_rs, sample_a_negative_peptide, group_peptides_by_parent_rs, encode_sequence_rs}, group_by_9mers};

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
            max_len,threshold,fold_neg,test_size);
    
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


fn prepare_data_for_seq_and_expression_data(data_tuple:(Vec<String>/* Peptide */,Vec<String>/* Allele name */,Vec<String>/* Tissue name */,Vec<u8> /* Label*/),
    pseudo_sequences:&HashMap<String,String> /* A hash map linking protein name to value */, max_len:usize /* maximum padding length */, 
    proteome:&HashMap<String,String>,
    anno_table:&HashMap<String, HashMap<String, ProteinInfo>>)->(
        (Array<u8,Ix2>/* Encode peptide sequences */,Array<u8,Ix2>/* Encode pseudo-sequences*/,
        Array<f32,Ix2>/* Encode gene expression*/,Array<u8,Ix2>/* Encoded labels*/)/* The encoded results from the array */, 
        
        (Vec<String>/* Peptide */,Vec<String>/* Allele name */,Vec<String>/* Tissue name */,Vec<u8> /* Label*/)
        /* The unmapped input */
    )
{
    let group_by_parent=group_peptides_by_parent_rs(&data_tuple.0, proteome); 
    // allocate vectors to hold the results 
    //----------
    // allocate vectors for holding the mapped data points 
    let num_dp=data_tuple.0.len()+1000; // an upper bound to avoid missing up with the data 
    let (mut peptides, mut pseudo_seq, mut gene_expression, mut labels)=(Vec::with_capacity(num_dp),
    Vec::with_capacity(num_dp), Vec::with_capacity(num_dp), Vec::with_capacity(num_dp)); // iterate over the elements of the array
    // allocate vectors for holding the unmapped data points 
    let (mut peptides_unmapped, mut pseudo_seq_unmapped, 
        mut tissue_unmapped, mut labels_unmapped)=(Vec::with_capacity(num_dp),
    Vec::with_capacity(num_dp), Vec::with_capacity(num_dp), Vec::with_capacity(num_dp)); // iterate over the elements of the array
    // Loop over all input data-points
    //--------------------------------
    for idx in 0..data_tuple.0.len()/*Loop over all the peptides*/
    {
        for protein in group_by_parent.get(&data_tuple.0[idx]).unwrap() /*Loop over all the parent of a peptide*/
        {
            match anno_table.get(&data_tuple.2[idx])
            {
                Some(tissue_table)=>
                {
                    match tissue_table.get(protein)
                    {
                        Some(protein_info)=>
                        {
                            peptides.push(data_tuple.0[idx].clone());
                            pseudo_seq.push(pseudo_sequences.get(&data_tuple.1[idx]).unwrap().clone());
                            gene_expression.push(protein_info.get_expression()); 
                            labels.push(data_tuple.3[idx].clone())
                        },
                        None=>
                        {
                            peptides_unmapped.push(data_tuple.0[idx].clone());
                            pseudo_seq_unmapped.push(data_tuple.1[idx].clone());
                            tissue_unmapped.push("UNK_tissue: ".to_string()+data_tuple.2[idx].as_str());
                            labels_unmapped.push(data_tuple.3[idx].clone());
                        }
                    }
                },
                None=>
                {
                    peptides_unmapped.push(data_tuple.0[idx].clone());
                    pseudo_seq_unmapped.push("UNK_allele: ".to_string()+data_tuple.1[idx].as_str());
                    tissue_unmapped.push(data_tuple.2[idx].clone());
                    labels_unmapped.push(data_tuple.3[idx].clone());
                }
            }
        }
    }
    // encode the results 
    //-------------------
    let encoded_peptide_seq=encode_sequence_rs(peptides,max_len); 
    let encoded_pseudo_seq=encode_sequence_rs(pseudo_seq,34);
    let encoded_gene_expression=Array::from_shape_vec((gene_expression.len(),1),gene_expression).unwrap();
    let encoded_labels=Array::from_shape_vec((labels.len(),1),labels).unwrap(); 

    // Return the results 
    //-------------------
    (
        (encoded_peptide_seq, encoded_pseudo_seq, encoded_gene_expression, encoded_labels),// define the training dataset 
        (peptides_unmapped, pseudo_seq_unmapped, tissue_unmapped, labels_unmapped)// define the test data sets
    )
}








pub fn generate_train_based_on_seq_exp_subcell<'py>(py:Python<'py>,
        input_seq:HashMap<String,Vec<String>>, path2cashedAnno_table:String, 
        tissue_name:String, path2pseudo:String, max_len:usize
    )->(
        (&'py PyArray<u8,Dim<[usize;2]>> /*Train pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train encoded sequence */,
         &'py PyArray<f32,Dim<[usize;2]>> /*Train expression values*/, &'py PyArray<f32,Dim<[usize;2]>> /*Train sub-cellular location values*/,
         &'py PyArray<f32,Dim<[usize;2]>> /* Train labels */),
        
        (&'py PyArray<u8,Dim<[usize;2]>> /*Test pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Test encoded sequence */,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test expression values*/, &'py PyArray<f32,Dim<[usize;2]>> /*Train sub-cellular location values*/,
        &'py PyArray<f32,Dim<[usize;2]>> /* Test labels */)
    )
{
    // Prepare the prelude for the encoders
    //------------------------------------- 
    let (train_data, test_data, pseudo_seq
        ,anno_table)=preparation_prelude(&input2prepare,&proteome,path2cashed_db,path2pseudo_seq,
        max_len,threshold,fold_neg,test_size);
    
    // encode and prepare the training data
    //-------------------------------------
    let (encoded_train_data,unmapped_train_data)=prepare_data_for_seq_and_label_data(train_data,
        &pseudo_seq,max_len,&proteome,&anno_table); 
    
    let(encoded_test_data,unmapped_test_data)=prepare_data_for_seq_and_label_data(test_data,
        &pseudo_seq,max_len,&proteome,&anno_table); 

    //return the results
    //------------------
    (
        (encoded_train_data.0.to_pyarray(py),encoded_train_data.1.to_pyarray(py),encoded_train_data.2.to_pyarray(py),ncoded_train_data.3.to_pyarray(py),encoded_train_data.4.to_pyarray(py)),
        (encoded_test_data.0.to_pyarray(py),encoded_test_data.1.to_pyarray(py),encoded_test_data.2.to_pyarray(py),encoded_test_data.3.to_pyarray(py)),
        unmapped_train_data,
        unmapped_test_data
    )
}

fn prepare_data_for_seq_exp_and_subcellular_data(data_tuple:(Vec<String>/* Peptide */,Vec<String>/* Allele name */,Vec<String>/* Tissue name */,Vec<u8> /* Label*/),
    pseudo_sequences:&HashMap<String,String> /* A hash map linking protein name to value */, max_len:usize /* maximum padding length */, 
    proteome:&HashMap<String,String>,
    anno_table:&HashMap<String, HashMap<String, ProteinInfo>>)->(
        (Array<u8,Ix2>/* Encode peptide sequences */,Array<u8,Ix2>/* Encode pseudo-sequences*/,
        Array<f32,Ix2>/* Encode gene expression*/,Array<u8,Ix2> /* Encode the subcellular locations*/,
        Array<u8,Ix2>/* Encoded labels*/)/* The encoded results from the array */, 
        
        (Vec<String>/* Peptide */,Vec<String>/* Allele name */,Vec<String>/* Tissue name */,Vec<u8> /* Label*/)
        /* The unmapped input */
    )
{
    let group_by_parent=group_peptides_by_parent_rs(&data_tuple.0, proteome); 
    // allocate vectors to hold the results 
    //----------
    // allocate vectors for holding the mapped data points 
    let num_dp=data_tuple.0.len()+1000; // an upper bound to avoid missing up with the peptides that happen in different proteins 
    let (mut peptides, mut pseudo_seq, mut gene_expression, mut subcellular_locations,
        mut labels)=(Vec::with_capacity(num_dp),
    Vec::with_capacity(num_dp), Vec::with_capacity(num_dp),Vec::with_capacity(num_dp), Vec::with_capacity(num_dp)); // iterate over the elements of the array
    // allocate vectors for holding the unmapped data points 
    let (mut peptides_unmapped, mut pseudo_seq_unmapped, 
        mut tissue_unmapped, mut labels_unmapped)=(Vec::with_capacity(num_dp),
    Vec::with_capacity(num_dp), Vec::with_capacity(num_dp), Vec::with_capacity(num_dp)); // iterate over the elements of the array
    // Loop over all input data-points
    //--------------------------------
    for idx in 0..data_tuple.0.len()/*Loop over all the peptides*/
    {
        for protein in group_by_parent.get(&data_tuple.0[idx]).unwrap() /*Loop over all the parent of a peptide*/
        {
            match anno_table.get(&data_tuple.2[idx])
            {
                Some(tissue_table)=>
                {
                    match tissue_table.get(protein)
                    {
                        Some(protein_info)=>
                        {
                            peptides.push(data_tuple.0[idx].clone());
                            pseudo_seq.push(pseudo_sequences.get(&data_tuple.1[idx]).unwrap().clone());
                            subcellular_locations.push(get_sub_cellular_location(protein_info.get_sub_cellular_loc())); // get the subcellular locations
                            gene_expression.push(protein_info.get_expression()); 
                            labels.push(data_tuple.3[idx].clone())
                        },
                        None=>
                        {
                            peptides_unmapped.push(data_tuple.0[idx].clone());
                            pseudo_seq_unmapped.push(data_tuple.1[idx].clone());
                            tissue_unmapped.push("UNK_tissue: ".to_string()+data_tuple.2[idx].as_str());
                            labels_unmapped.push(data_tuple.3[idx].clone());
                        }
                    }
                },
                None=>
                {
                    peptides_unmapped.push(data_tuple.0[idx].clone());
                    pseudo_seq_unmapped.push("UNK_allele: ".to_string()+data_tuple.1[idx].as_str());
                    tissue_unmapped.push(data_tuple.2[idx].clone());
                    labels_unmapped.push(data_tuple.3[idx].clone());
                }
            }
        }
    }
    // encode the results 
    //-------------------
    let encoded_peptide_seq=encode_sequence_rs(peptides,max_len); 
    let encoded_pseudo_seq=encode_sequence_rs(pseudo_seq,34);
    let encoded_gene_expression=Array::from_shape_vec((gene_expression.len(),1),gene_expression).unwrap();
    let encoded_subcellular_locations=Array::from_shape_vec((subcellular_location.len(), 1049),subcellular_location).unwrap();  
    let encoded_labels=Array::from_shape_vec((labels.len(),1),labels).unwrap(); 

    // Return the results 
    //-------------------
    (
        (encoded_peptide_seq, encoded_pseudo_seq, encoded_gene_expression, encoded_subcellular_locations,encoded_labels),// define the training dataset 
        (peptides_unmapped, pseudo_seq_unmapped, tissue_unmapped, labels_unmapped)// define the test data sets
    )
}



fn prepare_data_for_seq_sub_cell_location_exp_data(data_tuple:(Vec<String>/* Peptide */,Vec<String>/* Allele name */,Vec<String>/* Tissue name */,Vec<u8> /* Label*/),
    pseudo_sequences:&HashMap<String,String> /* A hash map linking protein name to value */, max_len:usize /* maximum padding length */, 
    proteome:&HashMap<String,String>,
    anno_table:&HashMap<String, HashMap<String, ProteinInfo>>)->(
        (Array<u8,Ix2>/* Encode peptide sequences */, Array<u8,Ix2>/* Encode pseudo-sequences*/,
        Array<f32,Ix2>/* Encode gene expression*/, Array<u8,Ix2>/* Encode subcellular locations*/,
        Array<u8,Ix2>/* Encoded labels*/)/* The encoded results from the array */, 
        
        (Vec<String>/* Peptide */,Vec<String>/* Allele name */,Vec<String>/* Tissue name */, Vec<String> /* subcellular location */, Vec<u8> /* Label*/)
        /* The unmapped input */
    )
{
    let group_by_parent=group_peptides_by_parent_rs(&data_tuple.0, proteome); 
    // allocate vectors to hold the results 
    //----------
    // allocate vectors for holding the mapped data points 
    let num_dp=data_tuple.0.len()+1000; // an upper bound to avoid missing up with the data 
    let (mut peptides, mut pseudo_seq, mut gene_expression, mut subcellular_location, mut labels)=(Vec::with_capacity(num_dp),
    Vec::with_capacity(num_dp), Vec::with_capacity(num_dp), Vec::with_capacity(num_dp)); // iterate over the elements of the array
    // allocate vectors for holding the unmapped data points 
    let (mut peptides_unmapped, mut pseudo_seq_unmapped, 
        mut tissue_unmapped, mut subcellular_location,mut labels_unmapped)=(Vec::with_capacity(num_dp),
    Vec::with_capacity(num_dp), Vec::with_capacity(num_dp), Vec::with_capacity(num_dp)); // iterate over the elements of the array
    // Loop over all input data-points
    //--------------------------------
    for idx in 0..data_tuple.0.len()/*Loop over all the peptides*/
    {
        for protein in group_by_parent.get(&data_tuple.0[idx]).unwrap() /*Loop over all the parent of a peptide*/
        {
            match anno_table.get(&data_tuple.2[idx])
            {
                Some(tissue_table)=>
                {
                    match tissue_table.get(protein)
                    {
                        Some(protein_info)=>
                        {
                            peptides.push(data_tuple.0[idx].clone());
                            pseudo_seq.push(pseudo_sequences.get(&data_tuple.1[idx]).unwrap().clone());
                            gene_expression.push(protein_info.get_expression()); 
                            labels.push(data_tuple.3[idx].clone())
                        },
                        None=>
                        {
                            peptides_unmapped.push(data_tuple.0[idx].clone());
                            pseudo_seq_unmapped.push(data_tuple.1[idx].clone());
                            tissue_unmapped.push("UNK_tissue: ".to_string()+data_tuple.2[idx].as_str());
                            labels_unmapped.push(data_tuple.3[idx].clone());
                        }
                    }
                },
                None=>
                {
                    peptides_unmapped.push(data_tuple.0[idx].clone());
                    pseudo_seq_unmapped.push("UNK_allele: ".to_string()+data_tuple.1[idx].as_str());
                    tissue_unmapped.push(data_tuple.2[idx].clone());
                    labels_unmapped.push(data_tuple.3[idx].clone());
                }
            }
        }
    }
    // encode the results 
    //-------------------
    let encoded_peptide_seq=encode_sequence_rs(peptides,max_len); 
    let encoded_pseudo_seq=encode_sequence_rs(pseudo_seq,34);
    let encoded_gene_expression=Array::from_shape_vec((gene_expression.len(),1),gene_expression).unwrap();
    let encoded_labels=Array::from_shape_vec((labels.len(),1),labels).unwrap(); 

    // Return the results 
    //-------------------
    (
        (encoded_peptide_seq, encoded_pseudo_seq, encoded_gene_expression, encoded_labels),// define the training dataset 
        (peptides_unmapped, pseudo_seq_unmapped, tissue_unmapped, labels_unmapped)// define the test data sets
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
         &'py PyArray<f32,Dim<[usize;2]>> /*Train context vectors*/,&'py PyArray<f32,Dim<[usize;2]>> /* Train labels */),
        
        (&'py PyArray<u8,Dim<[usize;2]>> /*Test pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Test encoded sequence */,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train sub-cellular location values*/,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test context vectors*/, &'py PyArray<f32,Dim<[usize;2]>> /* Test labels */)
    )
{
    // Prepare the prelude for the encoders
    //------------------------------------- 
    let (train_data, test_data, pseudo_seq
        ,anno_table)=preparation_prelude(&input2prepare,&proteome,path2cashed_db,path2pseudo_seq,
        max_len,threshold,fold_neg,test_size);
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

fn prepare_data_for_seq_exp_subcellular_and_context_data(data_tuple:(Vec<String>/* Peptide */,Vec<String>/* Allele name */,Vec<String>/* Tissue name */,Vec<u8> /* Label*/),
    pseudo_sequences:&HashMap<String,String> /* A hash map linking protein name to value */, max_len:usize /* maximum padding length */, 
    proteome:&HashMap<String,String>,
    anno_table:&HashMap<String, HashMap<String, ProteinInfo>>)->(
        (Array<u8,Ix2>/* Encode peptide sequences */,Array<u8,Ix2>/* Encode pseudo-sequences*/,
        Array<f32,Ix2>/* Encode gene expression*/,Array<u8,Ix2> /* Encode the subcellular locations*/,
        Array<u8,Ix2> /*Encode context vectors*/,Array<u8,Ix2>/* Encoded labels*/)/* The encoded results from the array */, 
        
        (Vec<String>/* Peptide */,Vec<String>/* Allele name */,Vec<String>/* Tissue name */,Vec<u8> /* Label*/)
        /* The unmapped input */
    )
{
    let group_by_parent=group_peptides_by_parent_rs(&data_tuple.0, proteome); 
    // allocate vectors to hold the results 
    //----------
    // allocate vectors for holding the mapped data points 
    let num_dp=data_tuple.0.len()+1000; // an upper bound to avoid missing up with the peptides that happen in different proteins 
    let (mut peptides, mut pseudo_seq, mut gene_expression, mut subcellular_locations,
       mut context_vectors, mut labels)=(Vec::with_capacity(num_dp),
    Vec::with_capacity(num_dp), Vec::with_capacity(num_dp),Vec::with_capacity(num_dp), Vec::with_capacity(num_dp),Vec::with_capacity(num_dp)); // iterate over the elements of the array
    // allocate vectors for holding the unmapped data points 
    let (mut peptides_unmapped, mut pseudo_seq_unmapped, 
        mut tissue_unmapped, mut labels_unmapped)=(Vec::with_capacity(num_dp),
    Vec::with_capacity(num_dp), Vec::with_capacity(num_dp), Vec::with_capacity(num_dp)); // iterate over the elements of the array
    // compute a hashmap between each vector 
    let context_vec_each_tissue = create_context_map(&anno_table);
    // Loop over all input data-points
    //--------------------------------
    for idx in 0..data_tuple.0.len()/*Loop over all the peptides*/
    {
        for protein in group_by_parent.get(&data_tuple.0[idx]).unwrap() /*Loop over all the parent of a peptide*/
        {
            match anno_table.get(&data_tuple.2[idx])
            {
                Some(tissue_table)=>
                {
                    match tissue_table.get(protein)
                    {
                        Some(protein_info)=>
                        {
                            context_vectors.push(context_vec_each_tissue.get(&data_tuple.2[idx]).unwrap().clone());
                            peptides.push(data_tuple.0[idx].clone());
                            pseudo_seq.push(pseudo_sequences.get(&data_tuple.1[idx]).unwrap().clone());
                            subcellular_locations.push(get_sub_cellular_location(protein_info.get_sub_cellular_loc())); // get the subcellular locations
                            gene_expression.push(protein_info.get_expression()); 
                            labels.push(data_tuple.3[idx].clone())
                        },
                        None=>
                        {
                            peptides_unmapped.push(data_tuple.0[idx].clone());
                            pseudo_seq_unmapped.push(data_tuple.1[idx].clone());
                            tissue_unmapped.push("UNK_tissue: ".to_string()+data_tuple.2[idx].as_str());
                            labels_unmapped.push(data_tuple.3[idx].clone());
                        }
                    }
                },
                None=>
                {
                    peptides_unmapped.push(data_tuple.0[idx].clone());
                    pseudo_seq_unmapped.push("UNK_allele: ".to_string()+data_tuple.1[idx].as_str());
                    tissue_unmapped.push(data_tuple.2[idx].clone());
                    labels_unmapped.push(data_tuple.3[idx].clone());
                }
            }
        }
    }
    // encode the results 
    //-------------------
    let encoded_peptide_seq=encode_sequence_rs(peptides,max_len); 
    let encoded_pseudo_seq=encode_sequence_rs(pseudo_seq,34);
    let encoded_gene_expression=Array::from_shape_vec((gene_expression.len(),1),gene_expression).unwrap();
    let encoded_subcellular_locations=Array::from_shape_vec((subcellular_locations.len(), 1049),subcellular_locations).unwrap();  
    let encoded_context_vector=Array::from_shape_vec((context_vectors.len(),context_vectors[0].len()), context_vectors).unwrap();
    let encoded_labels=Array::from_shape_vec((labels.len(),1),labels).unwrap(); 

    // Return the results 
    //-------------------
    (
        (encoded_peptide_seq, encoded_pseudo_seq, encoded_gene_expression, encoded_subcellular_locations,encoded_context_vector,encoded_labels),// define the training dataset 
        (peptides_unmapped, pseudo_seq_unmapped, tissue_unmapped, labels_unmapped)// define the test data sets
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
         &'py PyArray<f32,Dim<[usize;2]>> /* Train labels */),
        
        (&'py PyArray<u8,Dim<[usize;2]>> /*Test pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Test encoded sequence */,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Test sub-cellular location values*/,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test context vectors*/,&'py PyArray<u32,Dim<[usize;2]>> /*Test distance to glycosylation*/,
        &'py PyArray<f32,Dim<[usize;2]>> /* Test labels */), 

        (Vec<String>,Vec<String>,Vec<String>, Vec<u8>),// unmapped train data points 
        (Vec<String>,Vec<String>,Vec<String>, Vec<u8>) // unmapped test data points
    )
{
    // Prepare the prelude for the encoders
    //------------------------------------- 
    let (train_data, test_data, pseudo_seq
        ,anno_table)=preparation_prelude(&input2prepare,&proteome,path2cashed_db,path2pseudo_seq,
        max_len,threshold,fold_neg,test_size);
    
    // Prepare the encoder 



}

fn prepare_data_for_seq_exp_subcellular_context_d2g_data(data_tuple:(Vec<String>/* Peptide */,Vec<String>/* Allele name */,Vec<String>/* Tissue name */,Vec<u8> /* Label*/),
    pseudo_sequences:&HashMap<String,String> /* A hash map linking protein name to value */, max_len:usize /* maximum padding length */, 
    proteome:&HashMap<String,String>,
    anno_table:&HashMap<String, HashMap<String, ProteinInfo>>)->(
        (Array<u8,Ix2>/* Encode peptide sequences */,Array<u8,Ix2>/* Encode pseudo-sequences*/,
        Array<f32,Ix2>/* Encode gene expression*/,Array<u8,Ix2> /* Encode the subcellular locations*/,
        Array<u8,Ix2> /*Encode context vectors*/,Array<u8,Ix2> /* Encode the subcellular locations*/,
        Array<u8,Ix2>/* Encoded labels*/)/* The encoded results from the array */, 
        
        (Vec<String>/* Peptide */,Vec<String>/* Allele name */,Vec<String>/* Tissue name */,Vec<u8> /* Label*/)
        /* The unmapped input */
    )
{
    let group_by_parent=group_peptides_by_parent_rs(&data_tuple.0, proteome); 
    // allocate vectors to hold the results 
    //----------
    // allocate vectors for holding the mapped data points 
    let num_dp=data_tuple.0.len()+1000; // an upper bound to avoid missing up with the peptides that happen in different proteins 
    let (mut peptides, mut pseudo_seq, mut gene_expression, mut subcellular_locations,
       mut context_vectors, mut d2g, mut labels)=(Vec::with_capacity(num_dp),
    Vec::with_capacity(num_dp), Vec::with_capacity(num_dp),Vec::with_capacity(num_dp), Vec::with_capacity(num_dp),
    Vec::with_capacity(num_dp), Vec::with_capacity(num_dp)); // iterate over the elements of the array
    // allocate vectors for holding the unmapped data points 
    let (mut peptides_unmapped, mut pseudo_seq_unmapped, 
        mut tissue_unmapped, mut labels_unmapped)=(Vec::with_capacity(num_dp),
    Vec::with_capacity(num_dp), Vec::with_capacity(num_dp), Vec::with_capacity(num_dp)); // iterate over the elements of the array
    // compute a hashmap between each vector 
    let context_vec_each_tissue = create_context_map(&anno_table);
    // Loop over all input data-points
    //--------------------------------
    for idx in 0..data_tuple.0.len()/*Loop over all the peptides*/
    {
        for protein in group_by_parent.get(&data_tuple.0[idx]).unwrap() /*Loop over all the parent of a peptide*/
        {
            match anno_table.get(&data_tuple.2[idx])
            {
                Some(tissue_table)=>
                {
                    match tissue_table.get(protein)
                    {
                        Some(protein_info)=>
                        {
                            context_vectors.push(context_vec_each_tissue.get(&data_tuple.2[idx]).unwrap().clone());
                            peptides.push(data_tuple.0[idx].clone());
                            d2g.push(compute_nearest_distance_to_glycosylation(&data_tuple.0[idx],
                                &proteome.get(protein).unwrap(),protein_info.get_d2g()));
                            pseudo_seq.push(pseudo_sequences.get(&data_tuple.1[idx]).unwrap().clone());
                            subcellular_locations.push(get_sub_cellular_location(protein_info.get_sub_cellular_loc())); // get the subcellular locations
                            gene_expression.push(protein_info.get_expression()); 
                            labels.push(data_tuple.3[idx].clone())
                        },
                        None=>
                        {
                            peptides_unmapped.push(data_tuple.0[idx].clone());
                            pseudo_seq_unmapped.push(data_tuple.1[idx].clone());
                            tissue_unmapped.push("UNK_tissue: ".to_string()+data_tuple.2[idx].as_str());
                            labels_unmapped.push(data_tuple.3[idx].clone());
                        }
                    }
                },
                None=>
                {
                    peptides_unmapped.push(data_tuple.0[idx].clone());
                    pseudo_seq_unmapped.push("UNK_allele: ".to_string()+data_tuple.1[idx].as_str());
                    tissue_unmapped.push(data_tuple.2[idx].clone());
                    labels_unmapped.push(data_tuple.3[idx].clone());
                }
            }
        }
    }
    // encode the results 
    //-------------------
    let encoded_peptide_seq=encode_sequence_rs(peptides,max_len); 
    let encoded_pseudo_seq=encode_sequence_rs(pseudo_seq,34);
    let encoded_gene_expression=Array::from_shape_vec((gene_expression.len(),1),gene_expression).unwrap();
    let encoded_subcellular_locations=Array::from_shape_vec((subcellular_locations.len(), 1049),subcellular_locations).unwrap();  
    let encoded_context_vector=Array::from_shape_vec((context_vectors.len(),context_vectors[0].len()), context_vectors).unwrap();
    let encoded_distance_to_glyco=Array::from_shape_vec((d2g.len(),1), d2g).unwrap();
    let encoded_labels=Array::from_shape_vec((labels.len(),1),labels).unwrap(); 

    // Return the results 
    //-------------------
    (
        (encoded_peptide_seq, encoded_pseudo_seq, encoded_gene_expression, encoded_subcellular_locations,encoded_context_vector,encoded_distance_to_glyco,encoded_labels),// define the training dataset 
        (peptides_unmapped, pseudo_seq_unmapped, tissue_unmapped, labels_unmapped)// define the test data sets
    )
}

#[inline(always)]
fn compute_nearest_distance_to_glycosylation(peptide_seq:&String, protein_seq:&String, glyco_site:&String)->u32
{
    // compute the distance to glycosylation 
    //--------------------------------------
    let positions = protein_seq.match_indices(peptide_seq).map(|(index,_)|index as i32).collect::<Vec<_>>();
    let glyco_sites=glyco_site.split(";").map(|elem|elem.parse::<i32>().unwrap()).collect::<Vec<_>>();
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