/// loads a table linking protein IDs to ensemble ID
///--------------------------------------------------
/// 
/// 
 
use std::{path::Path, collections::HashMap}; 
use csv; 

/// parse the loaded peptide and make an ensemble figure out of it 
fn load_peptide_to_ensemble(path:&Path)->HashMap<String,String>// uniprot --> ENST
{
    csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(path)
        .unwrap().
        records().
        map(|record| 
            {
            let parsed_record=record.unwrap(); 
            (parsed_record[0].to_string(),parsed_record[1].to_string())
            })
        .collect::<HashMap<_,_>>()
}