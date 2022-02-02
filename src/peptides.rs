use std::{collections::{HashMap, HashSet}};
use rayon::prelude::*; 
use rand::{seq::SliceRandom, Rng};
use ndarray::{Array,Ix2};
use rand_distr::{Normal, Distribution}; 

/// ### Summary
/// Group peptides by the parent protein
/// ### Parameters
/// peptides: a vector of strings representing peptide sequences
/// proteome: a hashmap of protein names linked to protein sequences
/// ### Note 
/// searching is conducted against the VALUES of the hash map while the list returns the KEYS, i.e. the name of the parent peptide
/// ### Examples
/// ``` rust
/// use std::collections::{HashMap, HashSet}; // use some collections 
/// 
/// let peptides=vec!["PEPTIDE".to_string(),]; // an example peptide 
/// let mut proteome=HashMap::new(); // a Hashmap to be used for creating proteomes. 
/// 
///  // insert proteomes in the hash map
/// proteome.insert("PROT1".to_string(),"TEST_PEPTIDE_ONE".to_string()); 
/// proteome.insert("PROT2".to_string(),"TEST_PEPTIDE_TWO".to_string()); 
/// 
/// // group peptides by amino acid keys
/// let res=group_peptides_by_parent_rs(peptides,proteome); 
/// let mut expected_res=HashMap::new(); 
/// expected_res.insert("PEPTIDE".to_string(), vec!["TEST_PEPTIDE_ONE".to_string(),"TEST_PEPTIDE_TWO".to_string()]); 
/// assert!(expected_res,res); 
/// ```
pub fn group_peptides_by_parent_rs(peptides:Vec<String>,proteome:&HashMap<String,String>)->HashMap<String,Vec<String>>
{
    peptides
    .par_iter()
    .map(|peptide|
        {
            let parent=proteome
            .iter()
            .filter(|(_,seq)|seq.contains(&peptide))
            .map(|(name,_)|name.clone())
            .collect::<Vec<_>>();
            (peptide,parent)
        })
    .collect::<HashMap<_,_>>()
}

/// ### Summary
/// Encode a collection of sequences into an array of shape (num_seq,max_len)
/// ### parameters
/// input_seq: a vector of input strings representing the input sequences 
/// max_len: the maximum length of each sequences. longer sequences and trimmed while shorter sequences are padded
/// ### Examples
/// ```rust
///  use ndarray::prelude::*;
/// let seq=vec!["PEP".to_string(),"TEST".to_string(),"KLOPLLLLLLLL".to_owned()];
/// let array=encode_sequence_rs(seq,27);
/// println!("The generated results is: {:?}",array);
/// assert_eq!(array.shape(),ArrayView::from_shape((3,27),&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 7, 15,
/// 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 7, 16, 17,
/// 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 12, 24, 15, 12, 12, 12, 12, 12, 12, 12, 12]).unwrap().shape());
/// ```
pub fn encode_sequence_rs(input_seq:Vec<String>,max_len:usize)->Array<u8,Ix2>
{
    // encode elements to the elements into numbers
    let encoded_sequences=translate_seq_into_nums(&input_seq,max_len); // create a vector of u8 
    // stack the results into arrays
    let mut res_array=Vec::with_capacity(input_seq.len()*max_len);
    // loop over each array in the vector of arrays and fill the results
    for encoded_seq in encoded_sequences
    {
        res_array.extend_from_slice(&encoded_seq)
    }
    // create an array to hold the results
    Array::<u8,Ix2>::from_shape_vec((input_seq.len(), max_len ), res_array).unwrap()
}


#[inline(always)]
fn sample_a_negative_peptide(positive_peptides:&Vec<String>,proteome:&HashMap<String,String>)->String
{
    let mut sampler_rng=rand::thread_rng(); // create a RNG to sample the target proteins
    let mut position_sampler=rand::thread_rng(); // create a RNG to sample the position in the protein 
    let normal = Normal::new(15.0, 3.0).unwrap(); // create a normal distribution to sample the peptide from 
    // create the target proteins 
    let target_protein= unrolled_db.choose(&mut sampler_rng).unwrap(); // sample the protein
    let peptide_length= normal.sample(&mut rand::thread_rng()) as u32 ; // sample the peptide length from a normal distribution 
    peptide_length=std::cmp::min(21,std::cmp::max(9,v)); // clip the peptide length to be between [9,21]
    let position_in_backbone=position_sampler.gen_range(0..target_protein.1.len()-(peptide_length+1)); // sample the position in the protein backbone 
    let sampled_peptide=target_protein.1[position_in_backbone..position_in_backbone+peptide_length].to_string();
    // check that the peptide is not in positive peptides 
    if positive_peptides.contains(&sampled_peptide) // if it is there we try again
    {
        return sample_a_negative_peptide(positive_peptides,proteome)
    }
    else // if not we return a sampled peptide
    {
        return sampled_peptide
    }
}


pub fn generate_negative_by_sampling_rs(positive_peptides:Vec<String>, proteome:&HashMap<String,String>,fold_neg:u32)->(Vec<String>,Vec<u8>)
{
    // unroll the database for random sampling 
    //-----------------------------------------
    let unrolled_db= proteome
                .iter()
                .map(|(name,seq)|{(name.to_owned(),seq.to_owned())})
                .collect::<Vec<_>>();
    // set the number of sampled negatives
    let num_negatives=fold_neg*positive_peptides.len() as u32;
    // use a thread pool to generate the negatives 
    //--------------------------------------------
    let negative_peptides=(0..num_negatives)
        .into_par_iter() 
        .map(|_|{sample_a_negative_peptide(&positive_peptides,&proteome)})
        .collect::<Vec<String>>();
    // create the labels 
    //------------------
    let positive_labels=vec![1;positive_peptides.len()];
    let negative_labels=vec![0;negative_peptides.len()];
    //create the combine the positives and negatives and return the results
    //--------------------
    let mut sequences=Vec::with_capacity(positive_peptides.len()+negative_peptides.len()); 
    sequences.append(&mut positive_peptides);
    sequences.append(&mut negative_peptides);

    let mut labels=Vec::with_capacity(positive_labels.len()+negative_labels.len()); 
    labels.append(positive_labels); 
    labels.append(negative_labels); 

    // return the results
    //-------------------
    (sequences,labels)    
}





















/// ### Summary
/// A wrapper function for translating a vector of strings into a vector of vector of integers
/// ### Note
/// The function use a pool of threads to parallelize the execution across multiple threads 
/// ### Examples
/// ```rust
/// let seq=vec!["PEP".to_string(),"TEST".to_string(),"KLOPLLLLLLLL".to_owned()];
/// let res_vec=translate_seq_into_nums(seq,27);
/// assert_eq!(res_vec,vec![
/// vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 7, 15],
/// vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 7, 16, 17],
/// vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 12, 24, 15, 12, 12, 12, 12, 12, 12, 12, 12]);
/// ```

fn translate_seq_into_nums(peptides:&Vec<String>,max_len:usize)->Vec<Vec<u8>>
{
    peptides
    .par_iter()
    .map(|pep|translate_pep2num(pep,max_len))
    .collect::<Vec<_>>()
}

/// ### summary 
///
/// ### Note
/// 
/// ### Examples
/// ```rust
/// /// let seq=String::from("TESTONE");
/// let encoded_test=translate_pep2num(&seq,10);
/// assert_eq!(encoded_test,vec![0, 0, 0, 17, 7, 16, 17, 24, 3, 7]);
/// ```
#[inline(always)]
fn translate_pep2num(peptide:&String, max_len:usize)->Vec<u8>
{
    let effective_peptide; 
    if peptide.len()>max_len{effective_peptide=peptide[..max_len].to_string();}
    else{effective_peptide=peptide.to_string();}
    
    let mut encoded_seq=Vec::with_capacity(max_len as usize); 
    let num_zeros=max_len-effective_peptide.len();
    for _ in 0..num_zeros
    {
        encoded_seq.push(0);
    }
    // export the input sequence
    for elem in effective_peptide.chars()
    {
        match elem 
        {
            'A'=>encoded_seq.push(1),
            'R'=>encoded_seq.push(2),
            'N'=>encoded_seq.push(3),
            'D'=>encoded_seq.push(4),
            'C'=>encoded_seq.push(5),
            'Q'=>encoded_seq.push(6),
            'E'=>encoded_seq.push(7),
            'G'=>encoded_seq.push(8),
            'H'=>encoded_seq.push(10),
            'I'=>encoded_seq.push(11),
            'L'=>encoded_seq.push(12),
            'K'=>encoded_seq.push(13),
            'M'=>encoded_seq.push(14),
            'F'=>encoded_seq.push(15),
            'P'=>encoded_seq.push(15),
            'S'=>encoded_seq.push(16),
            'T'=>encoded_seq.push(17),
            'W'=>encoded_seq.push(18),
            'Y'=>encoded_seq.push(19),
            'V'=>encoded_seq.push(20),
            'B'=>encoded_seq.push(21),
            'U'=>encoded_seq.push(22),
            'X'=>encoded_seq.push(23),
            'O'=>encoded_seq.push(24),
            'Z'=>encoded_seq.push(25),
            'J'=>encoded_seq.push(26),
            _=>encoded_seq.push(26)
        }
    }
    encoded_seq
}


pub fn group_by_9mers_rs(peptides:Vec<String>)->HashMap<String,Vec<String>>
{
    get_unique_9_mers(&peptides)
    .into_par_iter()
    .map(|mer|
    {
        let peptides_belonging_to_mer=peptides
        .iter()
        .filter(|peptide|peptide.contains(&mer))
        .map(|pep|pep.to_owned())
        .collect::<Vec<_>>();

        (mer,peptides_belonging_to_mer)
    })
    .collect::<HashMap<_,_>>()
}
/// ### Samples
/// Generate a train ready dataset of input peptide sequences and labels, 
/// ### Notes negative examples are generated using random shuffling 
pub fn generate_a_train_db_by_shuffling_rs(mut positive_examples:Vec<String>, fold_neg:u32)->(Vec<String>,Vec<u8>)
{
    let mut labels_positive=vec![1 ;positive_examples.len()]; // create positive labels
    let mut negative_examples=generate_negative_by_shuffling_rs(&positive_examples,fold_neg); // create the negatives 
    let mut labels_neg=vec![0;negative_examples.len()]; // create the negative labels
    // know we need to concatenate the 
    let mut seq=Vec::with_capacity(positive_examples.len()+negative_examples.len()); 
    seq.append(&mut positive_examples); 
    seq.append(&mut negative_examples); 
    // return the labels
    let mut labels=Vec::with_capacity(labels_positive.len()+labels_neg.len()); 
    labels.append(&mut labels_positive); 
    labels.append(&mut labels_neg);
    // return the results
    //------------
    (seq,labels)
}


/// ### summary
/// A warper function used for generating negative examples from a collection of positive examples using a pool of threads
pub fn generate_negative_by_shuffling_rs(peptides:&Vec<String>, fold_neg:u32)->Vec<String>
{
    peptides
    .par_iter()
    .map(|peptide|shuffle_peptide_sequence(peptide,fold_neg))
    .flatten_iter()
    .collect::<Vec<_>>()
}

/// ### summary
/// shuffle the input peptide sequences and return a random collection of shuffled peptides
/// ### parameters
/// peptide: a string containing the peptide sequence 
/// fold_neg: the negative folds, i.e. number of negative examples from each positive example, where 1 is one negative for each 1 positive while 10 means 
/// 10 negatives for each positive example   
#[inline(always)]
pub fn shuffle_peptide_sequence(peptide:&String,fold_neg:u32)->Vec<String>
{
    // create a random number generator 
    let mut rng=rand::thread_rng(); 
    // allocate a vector to store the results
    let mut res=Vec::with_capacity(fold_neg as usize);
    // create random sequence form the input peptides 
    // create a counter for trying 
    let mut num_trials=fold_neg as i32;
    // loop to generate the negatives
    while num_trials!=0
    {
        // create a byte set from the input peptide
        let mut pep_as_bytes = peptide.clone().into_bytes();
        // translate the peptide into a bytes string 
        pep_as_bytes.shuffle(&mut rng);
        // create a random sequence from the shuffled peptide 
        let random_seq=String::from_utf8(pep_as_bytes.clone()).unwrap();
        // check that the results is in the generated random sequence
        if !res.contains(&random_seq)
        {
            res.push(String::from_utf8(pep_as_bytes).unwrap()); 
            num_trials-=1; 
        }
    }
    res
}
/// ### summary
/// get all unique 9 mers from a vector of peptides and returns a hash set 
#[inline(always)]
pub fn get_unique_9_mers(peptides:&Vec<String>)->HashSet<String>
{
    peptides
    .par_iter()
    .map(|peptide|fragment_peptide_into_9_mers(peptide))
    .flatten_iter()
    .collect::<HashSet<_>>()
}

/// ### summary
/// fragment a peptide into its unique 9-mers components
///  looping from 0->9, then 1->10 etc 
#[inline(always)]
pub fn fragment_peptide_into_9_mers(peptides:&String)->Vec<String>
{
    let mut results=Vec::with_capacity(peptides.len()-8); 
    for index in 0..(peptides.len()-8)
    {
        results.push(peptides[index..index+9].to_string())
    }
    results
}

#[cfg(test)]
mod test_peptides_function
{
    use super::*; 
    // loop over each function 

    #[test]
    fn analysis_encoding()
    {
        let seq=String::from("TESTONE");
        let encoded_test=translate_pep2num(&seq,10);
        assert_eq!(encoded_test,vec![0, 0, 0, 17, 7, 16, 17, 24, 3, 7]);
    }

    #[test]
    fn analysis_encoding_multiple()
    {
        use ndarray::prelude::*;
        let seq=vec!["PEP".to_string(),"TEST".to_string(),"KLOPLLLLLLLL".to_owned()];
        let array=encode_sequence_rs(seq,27);
        println!("The generated results is: {:?}",array);
        assert_eq!(array.shape(),ArrayView::from_shape((3,27),&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 7, 15,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 7, 16, 17,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 12, 24, 15, 12, 12, 12, 12, 12, 12, 12, 12]).unwrap().shape());
    }
}