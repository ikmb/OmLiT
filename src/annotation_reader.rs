/// A representation for the annotation-table class used for reading and generating multi-omic data about the protein structure
/// 
///
use std::{path::Path, collections::{HashSet, HashMap}}; 
use csv; 
use rayon::prelude::*; 
use crate::protein_info::*; 

/// A schematic representation for the annotation table which is a rust-sided code, 
/// The table is composite of the following structure:
/// 1. tissue which is the name of the tissue 
/// 2. protein_id which is the uniprot id for the input proteins
/// 3. nTPM which is the expression level of the parent transcript in the target tissue of interest 
/// 4. subCell_Loc which is a vector composite of the sub-cellular location of the target protein
/// 5. d2g which is the distance to glycosylation 
#[derive(Debug,Clone)]
pub struct AnnotationTable
{
    tissue:Vec<String>,
    protein_id:Vec<String>,
    nTPM:Vec<f32>,
    SubCell_Loc:Vec<String>,
    d2g:Vec<String>
}

impl AnnotationTable
{
    /// ### summary
    /// create a new instance of the new annotation table where the vectors of the table have not been allocated.
    fn new()->Self
    {
        AnnotationTable
        {
            tissue:Vec::new(),
            protein_id:Vec::new(),
            nTPM:Vec::new(),
            SubCell_Loc:Vec::new(),
            d2g:Vec::new()
        }
    }
    /// ### summary
    /// Create a table with a predefined capacity, 
    /// ### parameters
    /// num_rows: the number of rows in the table, if a None is used, a predefined value of 2,075,614 is used 
    fn with_capacity(num_rows:Option<usize>)->Self
    {
        let num_rows=match num_rows 
        {
            Some(num_rows) =>num_rows,
            None=>2_075_614, // the current number of entries in the database
        };
        AnnotationTable
        {
            tissue:Vec::with_capacity(num_rows),
            protein_id:Vec::with_capacity(num_rows),
            nTPM:Vec::with_capacity(num_rows),
            SubCell_Loc:Vec::with_capacity(num_rows),
            d2g:Vec::with_capacity(num_rows)
        }
    }
    /// ### summary
    /// push a data row to the annotation table
    /// ### parameter
    /// a StringRecord that have been generated from a default annotation table* (see project main webpage for more details)
    fn push(&mut self, row:&csv::StringRecord)->()
    {
        self.tissue.push(row[0].to_string());
        self.protein_id.push(row[1].to_string());
        self.nTPM.push(row[2].parse::<_>().unwrap()); 
        self.SubCell_Loc.push(row[3].to_string());
        self.d2g.push(row[4].to_string())
    }
    /// ### summary
    /// return a hashmap representation of the annotation table, the annotation map has the following representation 
    /// tissue_name --> HashMap:
    ///                     ProteinName --> ProteinInfo 
    ///                                         |__subcellular_location 
    ///                                         |__d2g
    ///                                         |__expression 
    /// ### Notes
    /// 1. The function use rayon to parallelize the execution 
    /// 2. for more information regard protin info struct check the protein_info.rs file 
    pub fn to_hashmap(&self)->HashMap<String,HashMap<String,ProteinInfo>>
    {
        self
            .get_unique_tissues()
            .into_par_iter()
            .map(|tissue_name|
                {
                    let tissue_results=self.get_index_of_tissues(&tissue_name)
                        .into_iter()
                        .map(|index|
                            {
                                let protein_name=self.protein_id[index].clone();
                                let protein_res=ProteinInfo::new(self.SubCell_Loc[index].clone(),
                                        self.d2g[index].clone(), self.nTPM[index]);
                                (protein_name,protein_res)                                 
                            })
                        .collect::<HashMap<_,_>>();
                    (tissue_name,tissue_results)
                })
            .collect::<HashMap<_,_>>()
    }
    /// ### Summary 
    /// reads an annotation table from a user provided table and return an AnnotationTable instance
    /// ### Parameters
    /// The path to the file where the file can be found, if reading the table failed the function panics !    
    pub fn read_table(path2file:&Path)->Self
    {
        // Create a reader 
        let mut reader=csv::ReaderBuilder::new().delimiter(b'\t').from_path(path2file).unwrap();
        
        // Create a empty table to be filled with the file content 
        let mut results=AnnotationTable::with_capacity(None); 

        // fill the file content into the table 
        for record in reader.records()
        {
            results.push(&record.unwrap())
        }
        // return the filed table
        results
    }
    /// ### Summary
    /// A helper function that is used for indexing the rows with the corresponding tissue name
    /// ### Parameters
    /// tissue --> which is a string containing the name  of the target tissue 
    /// ### returns 
    /// a vector of usizes where the tissue has been observed in the instance vectors
    fn get_index_of_tissues(&self, tissue:&String)->Vec<usize>
    {
        let mut results=Vec::with_capacity(16_606); // allocate a vector of indices  
        for index in 0..self.tissue.len() // loop over all entries and store the index of proteins that where select for binding to a protein
        {
            if self.tissue[index]==*tissue{results.push(index)}
        }
        results
    }
    /// ### Summary
    /// A summary function for collecting the unique tissues in an instance
    /// ### Returns
    /// a HashSet instance containing all unique tissues and cell lines in the database 
    fn get_unique_tissues(&self)->HashSet<String>
    {
        let mut unique_tissues=HashSet::new(); 
        for tissue in self.tissue.iter()
        {
            unique_tissues.insert(tissue.to_string()); 
        }
        unique_tissues
    }
}

#[cfg(test)]
mod testingAnnotationTable
{
    use super::*; 
    use std::time::Instant; 
    #[test]
    fn test_read_expression_table()
    {
        let path2file=Path::new("/Users/heshamelabd/projects/RustDB/Dev/final_assembled_database.tsv"); 
        let mut time=Vec::with_capacity(10);
        for _ in 0..5
        {
            let now= Instant::now(); 
            let csv_table=AnnotationTable::read_table(path2file); 
            time.push(now.elapsed().as_millis());
        }    
        let average=time.iter().sum::<u128>()/time.len() as u128;
        println!("######## The average time is: {}",average);
    }

    #[test]
    fn test_to_hasmap()
    {
        let path2file=Path::new("/Users/heshamelabd/projects/RustDB/Dev/final_assembled_database.tsv"); 
        let csv_table=AnnotationTable::read_table(path2file);
        let mut time=Vec::with_capacity(10);
        for _ in 0..5
        {
            let now= Instant::now(); 
            let csv_table=csv_table.to_hashmap(); 
            time.push(now.elapsed().as_millis());
        }    
        let average=time.iter().sum::<u128>()/time.len() as u128;
        println!("######## The average time is: {}",average);
    }
}