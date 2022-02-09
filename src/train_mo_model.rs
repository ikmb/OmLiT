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


fn preparation_prelude(input2prepare:&(Vec<String>,Vec<String>,Vec<String>),
proteome:&HashMap<String,String>, path2cashed_db:String, 
path2pseudo_seq:String, max_len:usize,
threshold:f32, fold_neg:u32, test_size:f32)->(
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



#[inline(always)]
fn group_by_allele_and_tissue(input2prepare:&(Vec<String>,Vec<String>,Vec<String>),)->HashMap<String,HashMap<String,Vec<String>>>
{
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

fn create_negative_database_from_positive(proteome:&HashMap<String,String>, 
        input2prepare:&(Vec<String>,Vec<String>,Vec<String>), 
        anno_table:&HashMap<String, HashMap<String, ProteinInfo>>, threshold:f32)->HashMap<String,Vec<String>>
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

#[inline(always)]
pub fn get_sub_cellular_location(go_terms:&String)->Vec<u8>
{
    let terms=go_terms.split(";")
            .map(|&elem| UNIQUE_GO_TERMS.iter().position(|unique_term|elem==unique_term).unwrap_or(1048))
            .collect::<Vec<>>(); 
    let mut results_vec=vec![0;1049];
    for term in terms
    {
        results_vec[term]=1
    }
    results_vec
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


/// ### Summary
/// Takes an annotation table as an input and return a hash-map containing tissue names as keys and context vectors as values
fn create_context_map(annoTable:&HashMap<String, HashMap<String, ProteinInfo>>)->HashMap<String,Vec<f32>>
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
            txp_exp.sort_by(|pair_1,pair_2|pair_1.1.cmp(&pair_2.1));
            
            // Create the context vector  
            let context_vector=txp_exp.into_iter().map(|(txp,exp)|exp).collect::<Vec<f32>>(); 

            (tissue_name.to_string(),context_vector)
        })
        .collect::<HashMap<_,_>>()
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
    let positions = protein_seq.match_indices(peptide_seq).map(|(index,_)|index as u32).collect::<Vec<_>>();
    let glyco_sites=glyco_site.split(";").map(|elem|elem.parse::<u32>().unwrap()).collect::<Vec<_>>();
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
                        match (*position..*position+peptide_seq.len() as u32).contains(site)
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