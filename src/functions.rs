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
/// 
/// 
#[inline]
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
        .filter(|(protein_name,protin_info)|protin_info.get_sub_cellular_loc()!="?")
        .collect::<HashMap<_,_>>())
}


pub fn annotate_proteins_in_tissues(target_protein:&Vec<(String,String)>,tissue_name:String,
    annoTable:&HashMap<String,HashMap<String,ProteinInfo>>)->HashMap<String,ProteinInfo>
{
    target_protein
        .par_iter()
        .filter(|(protein_name,tissue_name)|annoTable.contains_key(tissue_name))
        .map(|(protein_id,tissue_name)|
            {
                let target_protein_info=match annoTable.get(tissue_name).unwrap().get(protein_id)
                {
                    Some(res)=>res.clone(),
                    None=>ProteinInfo::new("?".to_string(),"?".to_string(),0.0) // Default protein then it can be filtered out in the end 
                }; 
                (protein_id.clone(),target_protein_info)
            })
            .filter(|(protein_name,protein_info)|protein_info.get_sub_cellular_loc()!="?")
            .collect::<HashMap<_,_>>()
}

pub fn annotate_all_proteins_in_one_tissue(path:&Path,
    target_protein:&Vec<String>,tissue_name:String)->Result<Vec<(String,String,String,f32)>,String>
{
    let database=AnnotationTable::read_table(path).to_hashmap();
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


pub fn cash_database_to_disk(path2file:&Path,path2res:&Path)->()
{
    // load the database
    let database=AnnotationTable::read_table(path2file).to_hashmap();
    // create a file to hold the results
    let mut f = BufWriter::new(File::create(path2res).unwrap());
    // serialize the results into the generated writer buffer 
    serialize_into(&mut f, &database).unwrap();
}

pub fn read_cashed_db(path2res:&Path)->HashMap<String,HashMap<String,ProteinInfo>>
{
    // open the file
    //--------------
    let file_reader = BufReader::new(File::open(path2res).unwrap());
    // load the serialized file 
    let res:HashMap<String,HashMap<String,ProteinInfo>>=deserialize_from(file_reader).unwrap(); 
    res
}
