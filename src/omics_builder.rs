/// The module contain pure rust functions used for preparing paired multi-omics model for training/retraining deep learning models on peptide HLA-II interaction 

/// Load the modules
use std::collections::HashMap;
use ndarray::{Array, Ix2};
use crate::{peptides::{encode_sequence_rs, group_peptides_by_parent_rs}, protein_info::ProteinInfo, utils::{get_sub_cellular_location, create_context_map, compute_nearest_distance_to_glycosylation}};


/// ### Summary 
/// prepare the data for training the model on the peptide sequence and the parent gene expression level 
/// ### Parameters 
/// data tuple: a tuple of four vectors containing the following elements
///     1. peptides: a vector of strings representing the peptide sequence 
///     2. allele names: a vector of strings containing the allele name 
///     3. tissue_name: a vector of strings containing the name of each tissue 
///     4. labels: a vector of u8 containing the label of each peptide, where 1 represent binders and 0 represent non-binder 
/// pseudo_sequences: a hashmap of HLA allele names (keys) and allele pseudo sequence (values)
/// proteome: a hashmap of protein names (keys) and the corresponding sequence (values)
/// anno_table: The annotation table, to extract information about the target proteins, in this case the expression level of the parent transcript in the target tissue.
/// ### Returns 
/// A tuple of 2 tuples, the first tuple is the encoded arrays tuple and has the following structure 
///     1. encoded peptide sequence --> an array of shape (num_mapped_peptides,max_len) and a type of u8 representing the encoded peptide sequences   
///     2. encoded pseudo sequence  --> an array of shape (num_mapped_peptides,34) and a type of u8 representing the encoded pseudo sequences   
///     3. encoded gene expression  --> an array of shape (num_mapped_peptides,1) and a type of float 32 representing the gene expression of the parent transcript.  
///     4. encoded labels --> an array of shape (num_mapped_peptides,1) and a type of float u8 representing the label of the peptide, where 1 represent binders, 0 represents non-binders
///  the second tuple, contain non-mapped peptides and has the same structure as the input tuple, i.e. data_tuple
pub fn prepare_data_for_seq_and_expression_data(data_tuple:(Vec<String>/* Peptide */,Vec<String>/* Allele name */,Vec<String>/* Tissue name */,Vec<u8> /* Label*/),
    pseudo_sequences:&HashMap<String,String> /* A hash map linking protein name to value */, max_len:usize /* maximum padding length */, 
    proteome:&HashMap<String,String>,
    anno_table:&HashMap<String, HashMap<String, ProteinInfo>>)->(
        /* the encoded data arrays, all have the same zero axis */
        (Array<u8,Ix2>/* Encode peptide sequences */,Array<u8,Ix2>/* Encode pseudo-sequences*/,
    Array<f32,Ix2>/* Encode gene expression*/,Array<u8,Ix2>/* Encoded labels*/)/* The encoded results from the array */, 
    
    /* The unmapped data points, either the allele name not in pseudo-sequence or the tissue name is not in the target database */
        (Vec<String>/* Peptide */,Vec<String>/* Allele name */,Vec<String>/* Tissue name */,Vec<u8> /* Label*/)
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

/// ### Summary 
/// prepare the data for training the model on the peptide sequence, the parent gene expression level and the subcellular location of the target peptide 
/// ### Parameters 
/// data tuple: a tuple of four vectors containing the following elements
///     1. peptides: a vector of strings representing the peptide sequence 
///     2. allele names: a vector of strings containing the allele name 
///     3. tissue_name: a vector of strings containing the name of each tissue 
///     4. labels: a vector of u8 containing the label of each peptide, where 1 represent binders and 0 represent non-binder 
/// pseudo_sequences: a hashmap of HLA allele names (keys) and allele pseudo sequence (values)
/// proteome: a hashmap of protein names (keys) and the corresponding sequence (values)
/// anno_table: The annotation table, to extract information about the target proteins, in this case the expression level of the parent transcript in the target tissue.
/// ### Returns 
/// A tuple of 2 tuples, the first tuple is the encoded arrays tuple and has the following structure 
///     1. encoded peptide sequence --> an array of shape (num_mapped_peptides,max_len) and a type of u8 representing the encoded peptide sequences   
///     2. encoded pseudo sequence  --> an array of shape (num_mapped_peptides,34) and a type of u8 representing the encoded pseudo sequences   
///     3. encoded gene expression  --> an array of shape (num_mapped_peptides,1) and a type of float 32 representing the gene expression of the parent transcript.  
///     4. encoded subcellular locations --> an array of shape (num_mapped_peptides, 1049) and a type of u8 representing the encoded protein array
///     4. encoded labels --> an array of shape (num_mapped_peptides,1) and a type of float u8 representing the label of the peptide, where 1 represent binders, 0 represents non-binders
///  the second tuple, contain non-mapped peptides and has the same structure as the input tuple, i.e. data_tuple
fn prepare_data_for_seq_exp_and_subcellular_data(data_tuple:(Vec<String>/* Peptide */,Vec<String>/* Allele name */,Vec<String>/* Tissue name */,Vec<u8> /* Label*/),
    pseudo_sequences:&HashMap<String,String> /* A hash map linking protein name to value */, max_len:usize /* maximum padding length */, 
    proteome:&HashMap<String,String>,
    anno_table:&HashMap<String, HashMap<String, ProteinInfo>>)->(
        /* the encoded data arrays, all have the same zero axis.*/    
        (Array<u8,Ix2>/* Encode peptide sequences */,Array<u8,Ix2>/* Encode pseudo-sequences*/,
        Array<f32,Ix2>/* Encode gene expression*/,Array<u8,Ix2> /* Encode the subcellular locations*/,
        Array<u8,Ix2>/* Encoded labels*/)/* The encoded results from the array */, 
        
        /* The unmapped input, containing the unmapped input tuples*/
        (Vec<String>/* Peptide */,Vec<String>/* Allele name */,Vec<String>/* Tissue name */,Vec<u8> /* Label*/)
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
    // unroll the subcellular data into a single array 
    //-------------------
    let num_dp=subcellular_locations.len();
    let mut subCellLocation_encoded_results=Vec::with_capacity(num_dp*1049); 
    for vec_code in subcellular_locations
    {
        subCellLocation_encoded_results.append(&mut vec_code)
    }
    // encode the data into arrays
    //-----------------------------
    let encoded_peptide_seq=encode_sequence_rs(peptides,max_len); 
    let encoded_pseudo_seq=encode_sequence_rs(pseudo_seq,34);
    let encoded_gene_expression=Array::from_shape_vec((gene_expression.len(),1),gene_expression).unwrap();
    let encoded_subcellular_locations=Array::from_shape_vec((num_dp, 1049),subCellLocation_encoded_results).unwrap();  
    let encoded_labels=Array::from_shape_vec((labels.len(),1),labels).unwrap(); 

    // Return the results 
    //-------------------
    (
        (encoded_peptide_seq, encoded_pseudo_seq, encoded_gene_expression, encoded_subcellular_locations,encoded_labels),// define the training dataset 
        (peptides_unmapped, pseudo_seq_unmapped, tissue_unmapped, labels_unmapped)// define the test data sets
    )
}

/// ### Summary 
/// prepare the data for training the model on the peptide sequences, the subcellular location and the context vector of the target peptide 
/// ### Parameters 
/// data tuple: a tuple of four vectors containing the following elements
///     1. peptides: a vector of strings representing the peptide sequence 
///     2. allele names: a vector of strings containing the allele name 
///     3. tissue_name: a vector of strings containing the name of each tissue 
///     4. labels: a vector of u8 containing the label of each peptide, where 1 represent binders and 0 represent non-binder 
/// pseudo_sequences: a hashmap of HLA allele names (keys) and allele pseudo sequence (values)
/// proteome: a hashmap of protein names (keys) and the corresponding sequence (values)
/// anno_table: The annotation table, to extract information about the target proteins, in this case the expression level of the parent transcript in the target tissue.
/// ### Returns 
/// A tuple of 2 tuples, the first tuple is the encoded arrays tuple and has the following structure 
///     1. encoded peptide sequence --> an array of shape (num_mapped_peptides,max_len) and a type of u8 representing the encoded peptide sequences   
///     2. encoded pseudo sequence  --> an array of shape (num_mapped_peptides,34) and a type of u8 representing the encoded pseudo sequences.   
///     3. encoded subcellular locations --> an array of shape (num_mapped_peptides, 1049) and a type of u8 representing the encoded protein array
///     4. encoded context vectors  --> an array of shape (num_mapped_peptides, num_genes_in_anno_table) and a type of f32 representing the expression of all genes in a specific cell or tissue 
///     5. encoded labels --> an array of shape (num_mapped_peptides,1) and a type of float u8 representing the labels of the peptide, where 1 represent binders, 0 represents non-binders
///  the second tuple, contain non-mapped peptides and has the same structure as the input tuple, i.e. data_tuple
fn prepare_data_for_seq_sub_cell_location_and_context_data(data_tuple:(Vec<String>/* Peptide */,Vec<String>/* Allele name */,Vec<String>/* Tissue name */,Vec<u8> /* Label*/),
    pseudo_sequences:&HashMap<String,String> /* A hash map linking protein name to value */, max_len:usize /* maximum padding length */, 
    proteome:&HashMap<String,String>,
    anno_table:&HashMap<String, HashMap<String, ProteinInfo>>)->(
        /* the encoded data arrays, all have the same zero axis.*/    
            (Array<u8,Ix2>/* Encode peptide sequences */, Array<u8,Ix2>/* Encode pseudo-sequences*/,
        Array<u8,Ix2>/* Encode subcellular locations*/, Array<f32,Ix2>/*context vectors*/,
        Array<u8,Ix2>/* Encoded labels*/)/* The encoded results from the array */, 
        
        /* The unmapped input, containing the unmapped input tuples*/
        (Vec<String>/* Peptide */,Vec<String>/* Allele name */,Vec<String>/* Tissue name */,Vec<u8> /* Label*/)
    )
{
    let group_by_parent=group_peptides_by_parent_rs(&data_tuple.0, proteome); 
    // allocate vectors to hold the results 
    //----------
    // allocate vectors for holding the mapped data points 
    let num_dp=data_tuple.0.len()+1000; // an upper bound to avoid missing up with the data 
    let (mut peptides, mut pseudo_seq, mut subcellular_locations, mut context_vectors, mut labels)=(Vec::with_capacity(num_dp),
    Vec::with_capacity(num_dp), Vec::with_capacity(num_dp), Vec::with_capacity(num_dp), Vec::with_capacity(num_dp)); // iterate over the elements of the array
    // allocate vectors for holding the unmapped data points 
    let (mut peptides_unmapped, mut pseudo_seq_unmapped, 
        mut tissue_unmapped, mut labels_unmapped)=(Vec::with_capacity(num_dp),
    Vec::with_capacity(num_dp), Vec::with_capacity(num_dp), Vec::with_capacity(num_dp)); 
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
                            peptides.push(data_tuple.0[idx].clone());
                            pseudo_seq.push(pseudo_sequences.get(&data_tuple.1[idx]).unwrap().clone());
                            context_vectors.push(context_vec_each_tissue.get(&data_tuple.2[idx]).unwrap().clone());
                            subcellular_locations.push(get_sub_cellular_location(protein_info.get_sub_cellular_loc())); // get the subcellular locations
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
    // let's unroll the compute vectors
    //---------------------------------
    // First annotate subcellular location 
    let mut subcellular_location_flatten=Vec::with_capacity(subcellular_locations.len()*1049);
    for vec in  subcellular_locations
    {
        subcellular_location_flatten.append(&mut vec); 
    }
    // second the context vectors
    //---------------------------
    let num_rows=context_vectors.len();
    let num_dp=context_vectors.iter().map(|vec|vec.len()).sum::<usize>(); 
    let mut context_vector_flatten=Vec::with_capacity(num_dp);
    for vec in  context_vectors
    {
        context_vector_flatten.append(&mut vec); 
    }
    // encode the results 
    let encoded_peptide_seq=encode_sequence_rs(peptides,max_len); 
    let encoded_pseudo_seq=encode_sequence_rs(pseudo_seq,34);
    let encoded_subcellular_location=Array::from_shape_vec((pseudo_seq.len(),1049),subcellular_location_flatten).unwrap();
    let encoded_context_vectors=Array::from_shape_vec((num_rows,(num_dp/num_rows) as usize),context_vector_flatten).unwrap();
    let encoded_labels=Array::from_shape_vec((labels.len(),1),labels).unwrap(); 

    // Return the results 
    //-------------------
    (
        (encoded_peptide_seq, encoded_pseudo_seq, encoded_subcellular_location, encoded_context_vectors, encoded_labels),// define the training dataset 
        (peptides_unmapped, pseudo_seq_unmapped, tissue_unmapped, labels_unmapped)// define the test data sets
    )
}

/// ### Summary 
/// prepare the data for training the model on the peptide sequences, the gene expression, the subcellular location and the context vector of the target peptide.  
/// ### Parameters 
/// data tuple: a tuple of four vectors containing the following elements
///     1. peptides: a vector of strings representing the peptide sequence 
///     2. allele names: a vector of strings containing the allele name 
///     3. tissue_name: a vector of strings containing the name of each tissue 
///     4. labels: a vector of u8 containing the label of each peptide, where 1 represent binders and 0 represent non-binder 
/// pseudo_sequences: a hashmap of HLA allele names (keys) and allele pseudo sequence (values)
/// proteome: a hashmap of protein names (keys) and the corresponding sequence (values)
/// anno_table: The annotation table, to extract information about the target proteins, in this case the expression level of the parent transcript in the target tissue.
/// ### Returns
/// A tuple of 2 tuples, the first tuple is the encoded arrays tuple and has the following structure 
/// A tuple of 2 tuples, the first tuple is the encoded arrays tuple and has the following structure 
///     1. encoded peptide sequence --> an array of shape (num_mapped_peptides,max_len) and a type of u8 representing the encoded peptide sequences   
///     2. encoded pseudo sequence  --> an array of shape (num_mapped_peptides,34) and a type of u8 representing the encoded pseudo sequences   
///     3. encoded gene expression  --> an array of shape (num_mapped_peptides,1) and a type of float 32 representing the gene expression of the parent transcript.
///     4. encoded subcellular locations --> an array of shape (num_mapped_peptides, 1049) and a type of u8 representing the encoded protein array
///     5. encoded context vectors  --> an array of shape (num_mapped_peptides, num_genes_in_anno_table) and a type of f32 representing the expression of all genes in a specific cell or tissue 
///     6. encoded labels --> an array of shape (num_mapped_peptides,1) and a type of float u8 representing the labels of the peptide, where 1 represent binders, 0 represents non-binders
/// 
fn prepare_data_for_seq_exp_subcellular_and_context_data(data_tuple:(Vec<String>/* Peptide */,Vec<String>/* Allele name */,Vec<String>/* Tissue name */,Vec<u8> /* Label*/),
    pseudo_sequences:&HashMap<String,String> /* A hash map linking protein name to value */, max_len:usize /* maximum padding length */, 
    proteome:&HashMap<String,String>,
    anno_table:&HashMap<String, HashMap<String, ProteinInfo>>)->(
        /* the encoded data arrays, all have the same zero axis.*/
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
    // let's unroll the compute vectors
    //---------------------------------
    // First annotate subcellular location 
    let mut subcellular_location_flatten=Vec::with_capacity(subcellular_locations.len()*1049);
    for vec in  subcellular_locations
    {
        subcellular_location_flatten.append(&mut vec); 
    }
    // second the context vectors
    //---------------------------
    let num_rows=context_vectors.len();
    let num_dp=context_vectors.iter().map(|vec|vec.len()).sum::<usize>(); 
    let mut context_vector_flatten=Vec::with_capacity(num_dp);
    for vec in  context_vectors
    {
        context_vector_flatten.append(&mut vec); 
    }
    
    // encode the results 
    //-------------------
    let encoded_peptide_seq=encode_sequence_rs(peptides,max_len); 
    let encoded_pseudo_seq=encode_sequence_rs(pseudo_seq,34);
    let encoded_gene_expression=Array::from_shape_vec((gene_expression.len(),1),gene_expression).unwrap();
    let encoded_subcellular_location=Array::from_shape_vec((pseudo_seq.len(),1049),subcellular_location_flatten).unwrap();
    let encoded_context_vectors=Array::from_shape_vec((num_rows,(num_dp/num_rows) as usize),context_vector_flatten).unwrap();
    let encoded_labels=Array::from_shape_vec((labels.len(),1),labels).unwrap(); 

    // Return the results 
    //-------------------
    (
        (encoded_peptide_seq, encoded_pseudo_seq, encoded_gene_expression, encoded_subcellular_location,encoded_context_vectors,encoded_labels),// define the training dataset 
        (peptides_unmapped, pseudo_seq_unmapped, tissue_unmapped, labels_unmapped)// define the test data sets
    )
}

/// ### Summary 
/// prepare the data for training the model on the peptide sequences, the gene expression, the subcellular location the context vector and the distance to glycosylation of the target peptide.  
/// ### Parameters 
/// data tuple: a tuple of four vectors containing the following elements
///     1. peptides: a vector of strings representing the peptide sequence 
///     2. allele names: a vector of strings containing the allele name 
///     3. tissue_name: a vector of strings containing the name of each tissue 
///     4. labels: a vector of u8 containing the label of each peptide, where 1 represent binders and 0 represent non-binder 
/// pseudo_sequences: a hashmap of HLA allele names (keys) and allele pseudo sequence (values)
/// proteome: a hashmap of protein names (keys) and the corresponding sequence (values)
/// anno_table: The annotation table, to extract information about the target proteins, in this case the expression level of the parent transcript in the target tissue.
/// ### Returns
/// A tuple of 2 tuples, the first tuple is the encoded arrays tuple and has the following structure 
/// A tuple of 2 tuples, the first tuple is the encoded arrays tuple and has the following structure 
///     1. encoded peptide sequence --> an array of shape (num_mapped_peptides,max_len) and a type of u8 representing the encoded peptide sequences   
///     2. encoded pseudo sequence  --> an array of shape (num_mapped_peptides,34) and a type of u8 representing the encoded pseudo sequences   
///     3. encoded gene expression  --> an array of shape (num_mapped_peptides,1) and a type of float 32 representing the gene expression of the parent transcript.
///     4. encoded subcellular locations --> an array of shape (num_mapped_peptides, 1049) and a type of u8 representing the encoded protein array
///     5. encoded context vectors  --> an array of shape (num_mapped_peptides, num_genes_in_anno_table) and a type of f32 representing the expression of all genes in a specific cell or tissue 
///     6. encoded labels --> an array of shape (num_mapped_peptides,1) and a type of float u8 representing the labels of the peptide, where 1 represent binders, 0 represents non-binders
fn prepare_data_for_seq_exp_subcellular_context_d2g_data(data_tuple:(Vec<String>/* Peptide */,Vec<String>/* Allele name */,Vec<String>/* Tissue name */,Vec<u8> /* Label*/),
    pseudo_sequences:&HashMap<String,String> /* A hash map linking protein name to value */, max_len:usize /* maximum padding length */, 
    proteome:&HashMap<String,String>,
    anno_table:&HashMap<String, HashMap<String, ProteinInfo>>)->(
        (Array<u8,Ix2>/* Encode peptide sequences */,Array<u8,Ix2>/* Encode pseudo-sequences*/,
        Array<f32,Ix2>/* Encode gene expression*/,Array<u8,Ix2> /* Encode the subcellular locations*/,
        Array<f32,Ix2> /*Encode context vectors*/,Array<u32,Ix2> /* Encode distance to glycosylation*/,
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
    // let's unroll the compute vectors
    //---------------------------------
    // First annotate subcellular location 
    let mut subcellular_location_flatten=Vec::with_capacity(subcellular_locations.len()*1049);
    for vec in  subcellular_locations
    {
        subcellular_location_flatten.append(&mut vec); 
    }
    // second the context vectors
    //---------------------------
    let num_rows=context_vectors.len();
    let num_dp=context_vectors.iter().map(|vec|vec.len()).sum::<usize>(); 
    let mut context_vector_flatten=Vec::with_capacity(num_dp);
    for vec in  context_vectors
    {
        context_vector_flatten.append(&mut vec); 
    }
    // encode the results 
    //-------------------
    let encoded_peptide_seq=encode_sequence_rs(peptides,max_len); 
    let encoded_pseudo_seq=encode_sequence_rs(pseudo_seq,34);
    let encoded_gene_expression=Array::from_shape_vec((gene_expression.len(),1),gene_expression).unwrap();
    let encoded_subcellular_location=Array::from_shape_vec((pseudo_seq.len(),1049),subcellular_location_flatten).unwrap();
    let encoded_context_vectors=Array::from_shape_vec((num_rows,(num_dp/num_rows) as usize),context_vector_flatten).unwrap();
    let encoded_distance_to_glyco=Array::from_shape_vec((d2g.len(),1), d2g).unwrap();
    let encoded_labels=Array::from_shape_vec((labels.len(),1),labels).unwrap(); 

    // Return the results 
    //-------------------
    (
        (encoded_peptide_seq, encoded_pseudo_seq, encoded_gene_expression, encoded_subcellular_location,encoded_context_vectors,encoded_distance_to_glyco,encoded_labels),// define the encoded training dataset 
        (peptides_unmapped, pseudo_seq_unmapped, tissue_unmapped, labels_unmapped) //  unmapped training data 
    )
}
