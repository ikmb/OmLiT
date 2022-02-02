/// The current module contain functions that are used for annotation and 
extern crate bincode;
use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;
use rayon::prelude::*; 
use crate::protein_info::*; 
use crate::annotation_reader::*; 
use std::fs::File;
use bincode::{serialize_into,deserialize_from};
use std::io::BufWriter;

/// ### Summary
/// Create proteins in a specific tissue
/// ### Parameters
/// target_protein: A vector of string representing protein names 
/// tissue_name: A string representing the tissue name 
/// annoTable: The annotation table database which is a hash map with the following structure
///     tissue_name --> HashMap:
///                     ProteinName --> ProteinInfo 
///                                         |__subcellular_location 
///                                         |__d2g
///                                         |__expression 
/// ### Notes
/// 1. The function will panic if the tissue name did not match any tissue defined in the database, also match is CASE-SENSITIVE
/// 2. The function uses Rayon to execute the performance across multiple threads
/// ### Returns
/// A hashmap with the following structure
///             String-->ProteinInfo
///                             |__subcellular_location 
///                             |__d2g
///                             |__expression
///  where the string is the name of the protein and the ProteinInfo is annotation for the provided protein in the target tissue  
#[inline(always)]
pub fn annotate_proteins_from_a_tissue(target_protein:&Vec<String>,tissue_name:String,
        annoTable:&HashMap<String,HashMap<String,ProteinInfo>>)->Result<HashMap<String,ProteinInfo>,String>
{
    let target_table=match annoTable.get(&tissue_name)
    {
        Some(table)=>table,
        None=>return Err(format!("The provided input tissue {}, is not in the annotation table.", tissue_name))
    };
    
    Ok(target_protein
        .par_iter()
        .map(|protein_id|
            {
                let target_protein_info=match target_table.get(protein_id)
                {
                    Some(res)=>res.clone(),
                    None=>ProteinInfo::new("?".to_string(),"?".to_string(),0.0) // Default protein then it can be filtered out in the end 
                }; 
                (protein_id.clone(),target_protein_info)
            })
        .filter(|(_,protin_info)|protin_info.get_sub_cellular_loc()!="?")
        .collect::<HashMap<_,_>>())
}

/// ### Summary
/// Annotate proteins in different tissues 
/// ### Parameters
/// 1. target_protein ==> A vector of tuple each is composite of two strings, the first is the protein-id and the second is the tissue name. 
///     Thus, the input structure is: tuple 1, tuple 2, ...., tuple n. Where each tuple is composite of:
///         a. protein_id --> a string representing protein-id
///         b. tissue_name --> a string representing the name of the tissue
/// 2. annoTable: The annotation table database which is a hash map with the following structure
///     tissue_name --> HashMap:
///                     ProteinName --> Tuple
///                                         |__ ProteinInfo
///                                         |    |__subcellular_location 
///                                         |    |__d2g
///                                         |__tissue_name
/// ### Returns
/// A hashmap with the following structure
///             String-->ProteinInfo
///                             |__subcellular_location 
///                             |__d2g
///                             |__expression
///  where the string is the name of the protein and the ProteinInfo is annotation for the provided protein in the target tissue, 
/// ### Note
/// 1. If the tissue name in one of the tuples is not defined, then the input tuple is skipped, e.g. vec![("PROT1".to_string(),"Tissue1".to_string()),
/// ("PROT2".to_string(),"Tissue2".to_string())] & if tissue two, i.e. Tissue2 is not in the database then the input tuple is cleared. 
/// 2. the function uses a pool of Rayon Threads to execute the code in parallel. 
pub fn annotate_proteins_in_tissues(target_protein:Vec<(String,String)>,
    annoTable:&HashMap<String,HashMap<String,ProteinInfo>>)->HashMap<String,(ProteinInfo,String)>
{
    target_protein
        .into_par_iter()
        .filter(|(_,tissue_name)|annoTable.contains_key(tissue_name))
        .map(|(protein_id,tissue_name)|
            {
                let target_protein_info=match annoTable.get(&tissue_name).unwrap().get(&protein_id)
                {
                    Some(res)=>res.clone(),
                    None=>ProteinInfo::new("?".to_string(),"?".to_string(),0.0) // Default protein then it can be filtered out in the end 
                }; 
                (protein_id.clone(),(target_protein_info,tissue_name))
            })
            .filter(|(_,(protein_info,_))|protein_info.get_sub_cellular_loc()!="?")
            .collect::<HashMap<_,_>>()
}

/// ### Summary
/// A convenient wrapper function used to automate the task of parsing the annotation table and generating the annotations for a collection of proteins 
/// in a specific tissue. 
/// ### Parameters
/// 1. path --> The path to load the annotation table 
/// 2. target_protein ---> The target proteins, which a vector of strings containing the name of proteins
/// 3. tissue --> a string representing the tissue name  
/// ### Returns 
/// A vector of tuples each of which is composite of 4 elements:
///     1. A string representing the protein name, i.e. uniprot ID 
///     1. A String representing the subcellular location
///     2. A String representing the distance to glycosylation 
///     3. A float representing the gene expression 
pub fn annotate_all_proteins_in_one_tissue(path:&Path,
    target_protein:&Vec<String>,tissue_name:String)->Vec<(String,String,String,f32)>
{
    let database=AnnotationTable::read_table(path).to_hashmap();
    annotate_proteins_from_a_tissue(target_protein,tissue_name,&database)
        .unwrap()
        .into_par_iter()
        .map(|(protein_name, protein_info)|
        {
            (protein_name,protein_info.get_sub_cellular_loc().clone(),protein_info.get_d2g().clone(),protein_info.get_expression())
        })
        .collect::<Vec<_>>()
}
/// ### Summary
/// A convenient wrapper function used to automate the task of parsing the annotation table and generating the annotations for a collection of proteins 
/// in a specific tissue. This used function used a pre-computed and cashed database for a faster execution speed. For more information check the function: 
/// cash_database_to_disk defined in the current modules.    
/// ### Parameters
/// 1. path --> The path to load the annotation table 
/// 2. target_protein ---> The target proteins, which a vector of strings containing the name of proteins
/// 3. tissue --> A string representing the tissue name  
/// ### Returns
/// A vector of tuples each of which is composite of 4 elements:
///     1. A string representing the protein name, i.e. uniprot ID 
///     1. A String representing the subcellular location
///     2. A String representing the distance to glycosylation 
///     3. A float representing the gene expression 
pub fn annotate_all_proteins_in_one_tissues_using_cashed_db(path2cashed_db:&Path,
    target_protein:&Vec<String>,tissue_name:String)->Result<Vec<(String,String,String,f32)>,String>
{
    let database=read_cashed_db(path2cashed_db); 
    Ok(annotate_proteins_from_a_tissue(target_protein,tissue_name,&database)
        .unwrap()
        .into_par_iter()
        .map(|(protein_name, protein_info)|
        {
            (protein_name,protein_info.get_sub_cellular_loc().clone(),protein_info.get_d2g().clone(),protein_info.get_expression())
        })
        .collect::<Vec<_>>()
    )
}


/// ### Summary
/// takes the path to an annotation table, parse and create an hashmap out of it finally it write the hashmap to the disk using serde. 
/// This enable the database to be reloaded relatively fast for subsequent usages. 
/// ### Parameters
///     1. path2file --> the path to the input annotation table 
///     2. path2res --> the pash to write the serialized hashmap 
pub fn cash_database_to_disk(path2file:&Path,path2res:&Path)->()
{
    // load the database
    let database=AnnotationTable::read_table(path2file).to_hashmap();
    // create a file to hold the results
    let mut file_writer = BufWriter::new(File::create(path2res).unwrap());
    // serialize the results into the generated writer buffer 
    serialize_into(&mut file_writer, &database).unwrap();
}
/// ### Summary 
/// Load the Serialized database from the disk
/// ### Parameters
/// path2res --> The path to cashed database, see the function cash_database_to_disk for more information 
pub fn read_cashed_db(path2res:&Path)->HashMap<String,HashMap<String,ProteinInfo>>
{
    // open the file
    //--------------
    let file_reader = BufReader::new(File::open(path2res).unwrap());
    // load the serialized file 
    let res:HashMap<String,HashMap<String,ProteinInfo>>=deserialize_from(file_reader).unwrap(); 
    res
}
