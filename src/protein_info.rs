/// ## summary
/// A simple representation for datasets 
/// 
use serde::{Serialize,Deserialize};

#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct ProteinInfo
{
    subCellLoc:String, 
    d2g:String, 
    expression:f32
}

impl ProteinInfo 
{
    pub fn new(subCellLoc:String, d2g:String, 
        expression:f32)->Self
    {
        ProteinInfo
        {
            subCellLoc, 
            d2g, 
            expression
        }
    }   
    #[inline]
    pub fn get_sub_cellular_loc(&self)->&String
    {
        &self.subCellLoc
    }
    
    #[inline]
    pub fn get_d2g(&self)->&String
    {
        &self.d2g
    }

    #[inline]
    pub fn get_expression(&self)->f32
    {
        self.expression
    }

    #[inline]
    pub fn own_sub_cellular_loc(self)->String
    {
        self.subCellLoc
    }

    #[inline]
    pub fn own_d2g(self)->String
    {
        self.d2g
    }
    //fn encode() ->(Vec<u8>,f32,f32)
}