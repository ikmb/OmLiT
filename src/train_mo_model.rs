use std::{collections::{HashMap, HashSet}, path::Path, hash::Hash, iter::FromIterator, ops::RangeBounds, alloc::LayoutErr};

use ndarray::{Dim, Array, Shape, Ix2};
use numpy::{PyArray, ToPyArray};
use pyo3::Python;
use rand::prelude::{SliceRandom, IteratorRandom};
use rayon::prelude::*; 

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
    let (encoded_train_data,unmapped_train_data)=prepare_data_for_seq_and_label_data(train_data,
        &pseudo_seq,max_len,&proteome,&anno_table); 
    
    let(encoded_test_data,unmapped_test_data)=prepare_data_for_seq_and_label_data(test_data,
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


fn prepare_data_for_seq_and_label_data(data_tuple:(Vec<String>/* Peptide */,Vec<String>/* Allele name */,Vec<String>/* Tissue name */,Vec<u8> /* Label*/),
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
fn sample_negatives_from_positive_data_structure(group_by_alleles:HashMap<String,HashMap<String,Vec<String>>>,
    target_proteomes:&HashMap<String,Vec<String>>, fold_neg:u32, test_size:f32,proteome:&HashMap<String,String>
    )->((Vec<String>/* peptide */,Vec<String>/* allele name */,Vec<String>/* tissue name */,Vec<u8> /* label*/),
        (Vec<String>/* peptide */,Vec<String>/* allele name */,Vec<String>/* tissue name */,Vec<u8> /* label*/)
    )
{
    // Check the test size is valid 
    if test_size<=0.0 || test_size>=1.0{panic!("In-valid test name: expected the test-size to be a float bigger than 0.0 and smaller than 1.0, however, the test size is: {}",test_size);}
    // Compute the size of the test dataset 
    //-------------------------------------
    let results:HashMap<String,HashMap<_,_>>=group_by_alleles
        .into_par_iter()
        .map(|(allele_name,tissue_to_peptides)|
        {
            let results_from_all_tissues=tissue_to_peptides
                .into_iter()
                .map(|(tissue_name,peptides)|
                {
                    
                    // group by the 9 mers cores
                    //-----------------------------------------------------
                    let grouped_by_peptides=group_by_9mers(peptides.clone()).unwrap();
                    let peptide_mers=grouped_by_peptides
                        .keys()
                        .map(|mers_core|mers_core.to_string())
                        .collect::<Vec<_>>(); 
                    
                    // Create the target tissue proteome to sample from 
                    //-------------------------------------------------
                    let target_proteins=target_proteomes.get(&tissue_name).unwrap();
                    let target_protein_seq=proteome
                            .iter()
                            .filter(|(name,seq)|target_proteins.contains(name))
                            .map(|(name,seq)|(name.clone().to_owned(),seq.clone().to_owned()))
                            .collect::<Vec<_>>();
                    
                    // Compute the size of the test dataset 
                    //-------------------------------------
                    let num_dp=grouped_by_peptides.len();
                    let num_test_dp=(num_dp as f32 *test_size) as usize; 
                    // create a random number generator
                    //--------------------- 
                    let mut rng=rand::thread_rng(); 
                    // prepare the train and test indices
                    //-----------------------------------
                    let test_index=(0..grouped_by_peptides.len())
                        .choose_multiple(&mut rng, num_test_dp);
                    
                    let train_index=(0..grouped_by_peptides.len())
                            .filter(|index|!test_index.contains(index))
                            .collect::<Vec<_>>(); 

                    
                    // prepare the train dataset
                    //--------------------------
                    // 1. positive peptides 
                    let mut positive_train_pep=train_index
                        .into_iter()
                        .map(|index|grouped_by_peptides.get(&peptide_mers[index]).unwrap())
                        .flatten()
                        .map(|pep|pep.clone().to_owned())
                        .collect::<Vec<_>>();
                    let mut positive_train_label=vec![1;positive_train_pep.len()]; 
                    
                    // 2. negative peptides
                    let mut sampled_negatives_train_pep=(0..positive_train_pep.len()*test_size as usize)
                        .into_iter()
                        .map(|_|sample_a_negative_peptide(&peptides,&target_protein_seq))
                        .collect::<Vec<String>>();
                    let mut sampled_negatives_train_label=vec![0;sampled_negatives_train_pep.len()]; 
                    
                    // 3. combine the results into a two vectors from sequences and labels
                    let mut train_seq=Vec::with_capacity(positive_train_pep.len()+sampled_negatives_train_pep.len()); 
                    train_seq.append(&mut positive_train_pep); 
                    train_seq.append(&mut sampled_negatives_train_pep); 

                    let mut train_labels=Vec::with_capacity(positive_train_label.len()+sampled_negatives_train_label.len());
                    train_labels.append(&mut positive_train_label); 
                    train_labels.append(&mut sampled_negatives_train_label);

                    // prepare the test dataset
                    //--------------------------
                    // 1. positive peptides 
                    let mut positive_test_pep=test_index
                        .into_iter()
                        .map(|index|grouped_by_peptides.get(&peptide_mers[index]).unwrap())
                        .flatten()
                        .map(|pep|pep.clone().to_owned())
                        .collect::<Vec<_>>();
                    let mut positive_test_label=vec![1;positive_test_pep.len()]; 
                    
                    // 2. negative peptides
                    let mut sampled_negatives_test_pep=(0..positive_test_pep.len()*test_size as usize)
                        .into_iter()
                        .map(|_|sample_a_negative_peptide(&peptides,&target_protein_seq))
                        .collect::<Vec<String>>();
                    let mut sampled_negatives_test_label=vec![0;sampled_negatives_test_pep.len()]; 
                    
                    // 3. combine the results into a two vectors from sequences and labels
                    let mut test_seq=Vec::with_capacity(positive_test_pep.len()+sampled_negatives_test_pep.len()); 
                    test_seq.append(&mut positive_test_pep); 
                    test_seq.append(&mut sampled_negatives_test_pep); 

                    let mut test_labels=Vec::with_capacity(positive_test_label.len()+sampled_negatives_test_label.len());
                    test_labels.append(&mut positive_test_label); 
                    test_labels.append(&mut sampled_negatives_test_label);
                    
                    // generate and return the output 
                    (tissue_name,
                        ((train_seq,train_labels),
                        (test_seq,test_labels))
                    )
                })
                .collect::<HashMap<String,((Vec<String>,Vec<u8>),(Vec<String>,Vec<u8>))>>(); 
                (allele_name.to_string(),results_from_all_tissues)
        })
        .collect::<HashMap<String,HashMap<String,((Vec<String>,Vec<u8>),(Vec<String>,Vec<u8>))>>>();
        // Unroll the results in to the vector form 
        //------------------------------------------
        // 1. get number of train and test data points 
        let num_train_dp=results.iter()
            .map(|(_,tissue_data)|
            {
                tissue_data.iter().map(|(_,(train_data,_))|train_data.0.len()).sum::<usize>()
            }) 
            .sum::<usize>(); 
        let num_test_dp=results.iter()
        .map(|(_,tissue_data)|
        {
            tissue_data.iter().map(|(_,(_,test_data))|test_data.0.len()).sum::<usize>()
        }) 
        .sum::<usize>(); 

        // 2. allocate vectors to hold the results 
        let (mut train_peptide, mut train_allele_name, mut train_tissue_name, mut train_label)=(
            Vec::with_capacity(num_train_dp),Vec::with_capacity(num_train_dp),Vec::with_capacity(num_train_dp),
        Vec::with_capacity(num_train_dp)); 

        let (mut test_peptide, mut test_allele_name, mut test_tissue_name, mut test_label)=(
            Vec::with_capacity(num_test_dp),Vec::with_capacity(num_test_dp),Vec::with_capacity(num_test_dp),
        Vec::with_capacity(num_test_dp));  

        // 3. un-roll the data into vectors
        for (allele_name,allele_data) in  results
        {
            for (tissue_name, (mut train_data, mut test_data)) in allele_data
            {
                // Prepare the training data
                //-------------------------- 
                train_allele_name.append(&mut vec![allele_name.clone();train_data.0.len()]); 
                train_tissue_name.append(&mut vec![tissue_name.clone();train_data.0.len()]); 
                train_peptide.append(&mut train_data.0); 
                train_label.append(&mut train_data.1); 
                
                // Prepare the test data
                //----------------------
                test_allele_name.append(&mut vec![allele_name.clone();test_data.0.len()]); 
                test_tissue_name.append(&mut vec![tissue_name.clone();test_data.0.len()]); 
                test_peptide.append(&mut test_data.0); 
                test_label.append(&mut test_data.1); 
            }
        }
        // Return the results to the encoder 
        //----------------------------------
        (
            (train_peptide,train_allele_name,train_tissue_name,train_label),
            (test_peptide,test_allele_name,test_tissue_name,test_label)
        )
        // Copy allele information 
        //------------------------
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
    
}

pub fn generate_train_based_on_seq_exp_subcell_context<'py>(py:Python<'py>,
        input_seq:HashMap<String,Vec<String>>, path2cashedAnno_table:String, 
        tissue_name:String, path2pseudo:String, max_len:usize
    )->(
        (&'py PyArray<u8,Dim<[usize;2]>> /*Train pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train encoded sequence */,
         &'py PyArray<f32,Dim<[usize;2]>> /*Train expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train sub-cellular location values*/,
         &'py PyArray<f32,Dim<[usize;2]>> /*Train context vectors*/,&'py PyArray<f32,Dim<[usize;2]>> /* Train labels */),
        
        (&'py PyArray<u8,Dim<[usize;2]>> /*Test pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Test encoded sequence */,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train sub-cellular location values*/,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test context vectors*/, &'py PyArray<f32,Dim<[usize;2]>> /* Test labels */)
    )
{
    
}

pub fn generate_train_based_on_seq_exp_subcell_context<'py>(py:Python<'py>,
        input_seq:HashMap<String,Vec<String>>, path2cashedAnno_table:String, 
        tissue_name:String, path2pseudo:String, max_len:usize
    )->(
        (&'py PyArray<u8,Dim<[usize;2]>> /*Train pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train encoded sequence */,
         &'py PyArray<f32,Dim<[usize;2]>> /*Train expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train sub-cellular location values*/,
         &'py PyArray<f32,Dim<[usize;2]>> /*Train context vectors*/,&'py PyArray<f32,Dim<[usize;2]>> /* Train labels */),
        
        (&'py PyArray<u8,Dim<[usize;2]>> /*Test pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Test encoded sequence */,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train sub-cellular location values*/,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test context vectors*/, &'py PyArray<f32,Dim<[usize;2]>> /* Test labels */)
    )
{
    
}

pub fn generate_train_based_on_seq_exp_context_d2g<'py>(py:Python<'py>,
        input_seq:HashMap<String,Vec<String>>, path2cashedAnno_table:String, 
        tissue_name:String, path2pseudo:String, max_len:usize
    )->(
        (&'py PyArray<u8,Dim<[usize;2]>> /*Train pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train encoded sequence */,
         &'py PyArray<f32,Dim<[usize;2]>> /*Train expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train sub-cellular location values*/,
         &'py PyArray<f32,Dim<[usize;2]>> /*Train distance context vectors*/, &'py PyArray<u32,Dim<[usize;2]>> /*Train distance to glycosylation*/,
         &'py PyArray<f32,Dim<[usize;2]>> /* Train labels */),
        
        (&'py PyArray<u8,Dim<[usize;2]>> /*Test pseudo-seq*/, &'py PyArray<u8,Dim<[usize;2]>> /*Test encoded sequence */,
        &'py PyArray<f32,Dim<[usize;2]>> /*Test expression values*/, &'py PyArray<u8,Dim<[usize;2]>> /*Train sub-cellular location values*/,
        &'py PyArray<f32,Dim<[usize;2]>> /*Train distance context vectors*/,&'py PyArray<u32,Dim<[usize;2]>> /*Test distance to glycosylation*/,
        &'py PyArray<f32,Dim<[usize;2]>> /* Test labels */)
    )
{
}
*/