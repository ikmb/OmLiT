/// Building an internal database to act as an API for getting gene expression 
/// of different dataset. 

use serde::Deserialize;
use std::{path::Path, collections::{HashMap, HashSet}}; 
use csv; 
use rayon::prelude::*; 

///
/// A Record type for encoding the data in each data-frame record 
#[derive(Clone,Debug,Deserialize)]
struct ExpressionTable
{
    gene:Vec<String>,
    tissue:Vec<String>,
    nTPM:Vec<f32>
}

impl  ExpressionTable 
{
    /// Create a new expression table 
    pub fn new()->Self
    {
        ExpressionTable
        {
            gene:Vec::new(),
            tissue:Vec::new(),
            nTPM:Vec::new(),
        }
    } 
    /// create a new expression table with the a known number of rows
    pub fn with_capacity(num_rows:Option<usize>)->Self
    {
        let num_rows= match num_rows
        {
            Some(num)=>num,
            None=>2510599 // the current number of entries in the expression database 
        }; 
        ExpressionTable
        {
            gene:Vec::with_capacity(num_rows),
            tissue:Vec::with_capacity(num_rows),
            nTPM:Vec::with_capacity(num_rows),
        }
    }
    /// push new data to the data frame
    fn push(&mut self, row: &csv::StringRecord)->()
    {
        self.gene.push(row[0].to_string());
        self.tissue.push(row[2].to_string());
        self.nTPM.push(row[3].parse::<f32>().unwrap());
    }
    /// build a nested hash-map 
    fn to_hashmap(&self)->HashMap<String,HashMap<String,f32>>
    {
        let unique_tissues=self.get_unique_tissues(); 
        let mut result_hashmap=HashMap::new(); 
        for tissue in unique_tissues
        {
            // get the index of where element is
            let target_indices=self.get_index(0, &tissue).unwrap(); 
            let mut target_ds=HashMap::new();
            for index in 0..target_indices.len()
            {
                if target_indices[index]==1
                {
                    target_ds.insert(self.gene[index].to_string(), self.nTPM[index]);
                }
            }
            // add the results to results
            result_hashmap.insert(tissue, target_ds);
        }
        result_hashmap
    }
    fn to_hashmap_parallel(&self)->HashMap<String,HashMap<String,f32>>
    {
        let unique_tissues=self.get_unique_tissues(); 

        unique_tissues
                .into_par_iter()
                .map(|tissue_name|
                    {
                        let target_indices=self.get_index(0, &tissue_name).unwrap(); 
                        let mut target_ds=HashMap::new();
                        for index in 0..target_indices.len()
                        {
                            if target_indices[index]==1
                            {
                                target_ds.insert(self.gene[index].to_string(), self.nTPM[index]);
                            }
                        }
                        (tissue_name,target_ds)
                    })
                .collect::<HashMap<_,_>>()
    }

    /// get the index of all genes belonging to a specific tissue or manuscript 
    fn get_index(&self, column:i8, value:&String)->Result<Vec<usize>,String>
    {
        match column
        {
            0=>
            {
                let mut index=vec![0;self.tissue.len()];
                for idx in 0..self.tissue.len()
                {
                    if self.tissue[idx]==*value
                    {
                        index[idx]=1
                    }
                }
                Ok(index)
            },
            1=>
            {
                let mut index=vec![0;self.gene.len()]; 
                for idx in 0..self.gene.len()
                {
                    if self.tissue[idx]==*value
                    {
                        index[idx]=1
                    }
                }
                Ok(index)
            },
            _=>Err(format!("Not supported!, only indexing along the first and second axis is supported"))
        }
    }
    /// get the unique tissue in the database 
    fn get_unique_tissues(&self)-> HashSet<String>
    {
        let mut unique_tissue=HashSet::new(); 
        for tissue_name in self.tissue.iter()
        {
            unique_tissue.insert(tissue_name.to_string());
        }
        unique_tissue
    }
    /// get the unique transcripts in the database
    fn get_unique_transcripts(&self)-> HashSet<String>
    {
        let mut unique_transcript=HashSet::new(); 
        for transcript in self.tissue.iter()
        {
            unique_transcript.insert(transcript.to_string());
        }
        unique_transcript
    }
    /// read the input database
    pub fn read_tsv(path2file:&Path, num_rows:Option<usize>)->Result<ExpressionTable,String>
    {
        // our file reader 
        let mut reader= match csv::ReaderBuilder::new().delimiter(b'\t').from_path(path2file)
        {
            Ok(file)=>file,
            Err(_)=> return Err(format!("Reading the input file: {:?} does not exists", path2file))    
        };
        // create a new data frame 
        let mut read_data=ExpressionTable::with_capacity(num_rows); 
        // push the results into a data frame
        for record in reader.records()
        {
            read_data.push(&record.unwrap())
        } 
        // return the read results 
        Ok(read_data)
    }
}

#[cfg(test)]
mod test_reader
{
    // load the modules and the dataset 
    use std::path::Path;
    use  super::*; 
    use std::time::Instant; 
    
    #[test]
    fn test_read_expression_table()
    {
        let path2file=Path::new("/Users/heshamelabd/projects/RustDB/Dev/Expression_reference_database_full.tsv"); 
        let mut time=Vec::with_capacity(10);
        for _ in 0..5
        {
            let now= Instant::now(); 
            let csv_table=ExpressionTable::read_tsv(path2file, None); 
            time.push(now.elapsed().as_millis());
        }    
        let average=time.iter().sum::<u128>()/time.len() as u128;
        println!("######## The average time is: {}",average);
    }

    #[test]
    fn test_to_hashmap()
    {
        let path2file=Path::new("/Users/heshamelabd/projects/RustDB/Dev/dev_ds_2.tsv"); 
        let csv_table=ExpressionTable::read_tsv(path2file, None).unwrap(); 
        let mut time=Vec::with_capacity(10);
        for _ in 0..10
        {
            let now= Instant::now(); 
            let res=csv_table.to_hashmap_parallel(); 
            time.push(now.elapsed().as_millis());
        }   
        let average=time.iter().sum::<u128>()/time.len() as u128;
        println!("######## The average time for parallel execution is: {}",average);
        time.clear();
        for _ in 0..10
        {
            let now= Instant::now(); 
            let res=csv_table.to_hashmap(); 
            time.push(now.elapsed().as_millis());
        }   
        let average=time.iter().sum::<u128>()/time.len() as u128;
        println!("######## The average time for single execution is: {}",average);
    }
}