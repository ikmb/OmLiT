/// Provides a wrapper for working and loading glycosylation data
/// 

use std::{path::Path, collections::HashMap}; 
use csv; 

pub fn read_glyco_table(path:&Path)->HashMap<String,Vec<u16>>
{
    // create a reader to the file 
    let mut file_reader=csv::ReaderBuilder::new().delimiter(b'\t').from_path(path).unwrap(); 
    
    // create a hashmap to hold the results
    let mut results=HashMap::new();

    // file the map
    for record in file_reader.records()
    {
       let record_parsed=record.unwrap(); 
       let res;  
       if record_parsed[1].contains(';')
       {
            res=record_parsed[1]
                            .split(';')
                            .map(|num|num.parse::<u16>().unwrap())
                            .collect::<Vec<_>>();
       }
       else
       {
            res=vec![record_parsed[1].parse::<u16>().unwrap();1];   
       }
       results.insert(record_parsed[0].to_string(), res); 
    } 
    results
}
