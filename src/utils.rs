use std::collections::HashMap;
use csv; 

/// ### Summary 
/// A reader function for reading HLA pseudo sequences
/// ### Parameters 
/// The path to the file containing the pseudo sequences
/// ### Return
/// returns a hashmap of allele names to protein sequences
/// ---------------------------
pub fn read_pseudo_seq(input2file:&Path)->HashMap<String,String>
{
    let file_reader=csv::ReaderBuilder::new().delimiter(b'\t').from_path(input2file).unwrap(); 
    let mut res_map=HashMap::new();

    for record in file_reader.records()
    {
        let row=record.unwrap();
        res_map.insert(record[0], record[1]);
    }
    res_map
}
