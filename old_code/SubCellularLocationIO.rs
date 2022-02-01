/// 
/// 
/// 
use std::{path::Path, collections::{HashMap, HashSet}}; 
use rayon::prelude::*; 
use csv; 

#[derive(Clone,Debug)]
struct SCLTable
{
    protein_id:Vec<String>,
    go_term:Vec<String>
}

impl SCLTable
{
    /// create a new subcellular locations 
    fn new()->Self
    {
        SCLTable
        {
            protein_id:Vec::new(),
            go_term:Vec::new()
        }
    }
    /// create a new table with predetermined capacity 
    fn with_capacity(num_prot:Option<usize>)->Self
    {
        // get the number of rows form the input single cell table 
        let num_rows=match num_prot
        {
            Some(num)=>num, 
            None=>142556 // Default value for the number of rows in the table 
        };
        SCLTable
        {
            protein_id:Vec::with_capacity(num_rows),
            go_term:Vec::with_capacity(num_rows),
        }
    }
    
    /// read the provided table 
    fn read_table(path:&Path)->Self
    {
        // load the reader 
        let mut reader=csv::ReaderBuilder::new().delimiter(b'\t').from_path(path).unwrap();

        let mut scl_table=SCLTable::with_capacity(None);

        for record in reader.records()
        {
            scl_table.push(&record.unwrap())
        }
        scl_table
 
    }
    /// pushes update to the database 
    fn push(&mut self, row:&csv::StringRecord)->()
    {
        self.protein_id.push(row[1].to_string());
        self.go_term.push(row[4].to_string());
    }
    /// reshape the table into hashmaps
    fn to_hashmap(&self)->HashMap<String,Vec<String>>
    {
        let unique_proteins=self.get_unique_proteins();
        let mut results_all_proteins=HashMap::new(); 
        // get pep in proteins 
        for protein in unique_proteins.into_iter()
        {
            // get the index of target protein
            let index_target_protein=self.get_indices(&protein);
            // allocate a vector to hold the results
            let mut GO_terms = Vec::with_capacity(index_target_protein.len());
            // copy the results from the table
            for idx in index_target_protein
            {
                GO_terms.push(self.go_term[idx].clone())
            }
            // Add the results
            //----------------
            results_all_proteins.insert(protein,GO_terms);
        }
        results_all_proteins
    }
    /// reshape the table into hashmaps using a pool of threads 
    fn to_hashmap_parallel(&self)->HashMap<String,Vec<String>>
    {
        let unique_proteins=self.get_unique_proteins();
        // get pep in proteins 
        unique_proteins
                .into_par_iter()
                .map(|unique_protein|
                    {
                        // get the index of target protein
                        let index_target_protein=self.get_indices(&unique_protein);
                        // allocate a vector to hold the results
                        let mut GO_terms = Vec::with_capacity(index_target_protein.len());
                        // copy the results from the table
                        for idx in index_target_protein
                        {
                            GO_terms.push(self.go_term[idx].clone())
                        }
                        (unique_protein,GO_terms)
                    })
                .collect::<HashMap<_,_>>()
    }
    /// Return the indices of the provided protein in the target collection of proteins
    fn get_indices(&self,protein_id:&String)->Vec<usize>
    {
        // Loop over results
        let mut results=Vec::with_capacity(100); 
        // loop over all proteins
        for idx in 0..protein_id.len()
        {
            if self.protein_id[idx]==*protein_id
            {
                results.push(idx)
            }
        }
        results
    }
    /// get unique protein from the database 
    fn get_unique_proteins(&self)->HashSet<String>
    {
        let mut unique_proteins=HashSet::new(); 
        for protein in self.protein_id.iter()
        {
            unique_proteins.insert(protein.to_string());
        }
        unique_proteins
    }
    fn len(&self)->usize
    {
        self.protein_id.len()
    }
    fn get(&self, index:usize)->(&String,&String)
    {
        (&self.protein_id[index],&self.go_term[index])
    }
}

#[cfg(test)]
mod test_SCL
{
    use super::*; 
    use std::time::Instant; 

    #[test]
    fn test_subcellular_location()
    {
        let path= Path::new("/Users/heshamelabd/projects/RustDB/Dev/GO_location.tsv"); 
        let dev_case=SCLTable::read_table(&path); 
        for idx in 0..10
        {
            let (protein, go_term)=dev_case.get(idx); 
            println!("protein: {}, Go: {}",protein,go_term)
        }
        assert_eq!(1,0)
    }
    #[test]
    fn test_to_hashmap()
    {
        let path= Path::new("/Users/heshamelabd/projects/RustDB/Dev/GO_location.tsv"); 
        let dev_case=SCLTable::read_table(&path); 
        
        for idx in 0..10
        {
            
        }
        assert_eq!(1,0)
    }
    #[test]
    fn test_to_hashmap_parallel()
    {
        let path= Path::new("/Users/heshamelabd/projects/RustDB/Dev/GO_location.tsv"); 
        let dev_case=SCLTable::read_table(&path); 
        let mut time=Vec::with_capacity(10);
        for _ in 0..10
        {
            let now= Instant::now(); 
            let res=dev_case.to_hashmap_parallel(); 
            time.push(now.elapsed().as_millis());
        }   
        let average=time.iter().sum::<u128>()/time.len() as u128;
        println!("######## The average time for parallel execution is: {}",average);
        time.clear();
        for _ in 0..10
        {
            let now= Instant::now(); 
            let res=dev_case.to_hashmap(); 
            time.push(now.elapsed().as_millis());
        }   
        let average=time.iter().sum::<u128>()/time.len() as u128;
        println!("######## The average time for single execution is: {}",average);
    }
}



