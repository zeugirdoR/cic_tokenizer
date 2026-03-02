pub mod geometry;

use pyo3::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;
use statrs::function::gamma::ln_gamma;
use nalgebra::Vector3;

use crate::geometry::TokenizerNode;

// ==========================================
// 1. CONTINUOUS GEOMETRY ENGINE (RICCI FLOW)
// ==========================================

pub struct BatchCounts {
    pub total_n: f64,
    pub n1: f64,  
    pub n0: f64,  
    pub n01: f64, 
    pub n00: f64, 
    pub n11: f64, 
    pub n10: f64, 
}

impl BatchCounts {
    pub fn compute_score_u(&self, theta: &Vector3<f64>) -> Vector3<f64> {
        let eps = 1e-12;
        let t1 = theta[0];
        let t2_given_0 = theta[1];
        let t2_given_1 = theta[2];

        let u1 = (self.n1 / (t1 + eps)) - (self.n0 / (1.0 - t1 + eps));
        let u2_0 = (self.n01 / (t2_given_0 + eps)) - (self.n00 / (1.0 - t2_given_0 + eps));
        let u2_1 = (self.n11 / (t2_given_1 + eps)) - (self.n10 / (1.0 - t2_given_1 + eps));

        Vector3::new(u1, u2_0, u2_1)
    }
}

pub fn process_batch(
    network: &mut HashMap<String, TokenizerNode>, 
    pair_key: &str, 
    counts: BatchCounts, 
    kappa: f64
) {
    let node = network.entry(pair_key.to_string()).or_insert_with(|| TokenizerNode {
        theta: Vector3::new(0.5, 0.5, 0.5), 
        n_samples: 0.0,
    });

    node.n_samples += counts.total_n;
    let score_u = counts.compute_score_u(&node.theta);
    let grad_r = Vector3::zeros(); 

    node.geometric_step(score_u, grad_r, kappa);
}

pub fn stream_text_to_geometry(
    text: &str,
    network: &mut HashMap<String, TokenizerNode>,
    kappa: f64
) {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 2 { return; }

    let total_n = (chars.len() - 1) as f64;
    let mut marginals: HashMap<char, f64> = HashMap::new();
    let mut joints: HashMap<(char, char), f64> = HashMap::new();

    for i in 0..chars.len() - 1 {
        let a = chars[i];
        let b = chars[i+1];
        *marginals.entry(a).or_insert(0.0) += 1.0;
        *joints.entry((a, b)).or_insert(0.0) += 1.0;
    }
    *marginals.entry(chars[chars.len() - 1]).or_insert(0.0) += 1.0;

    for (&(a, b), &n11) in joints.iter() {
        let n1 = *marginals.get(&a).unwrap_or(&0.0);
        let n_b_total = *marginals.get(&b).unwrap_or(&0.0);

        let counts = BatchCounts {
            total_n,
            n1,                                    
            n0: total_n - n1,                      
            n11,                                   
            n10: n1 - n11,                         
            n01: n_b_total - n11,                  
            n00: total_n - n1 - (n_b_total - n11), 
        };

        let pair_key = format!("{}{}", a, b);
        process_batch(network, &pair_key, counts, kappa);
    }
}

// ==========================================
// 2. DISCRETE EXACT CIC TOKENIZER (PYO3)
// ==========================================

#[derive(Hash, Eq, PartialEq, Clone, Copy)]
struct TokenPair(u32, u32);

#[pyclass]
pub struct CICTokenizer {
    sequence: Vec<u32>,
    vocab: HashMap<u32, Vec<u8>>,
    counts: HashMap<u32, usize>,
    next_token_id: u32,
}

#[inline(always)]
fn x_ln_x(x: f64) -> f64 {
    if x <= 0.0 { 0.0 } else { x * x.ln() }
}

#[pymethods]
impl CICTokenizer {
    #[new]
    pub fn new(text: &str) -> Self {
        let bytes = text.as_bytes();
        let mut sequence = Vec::with_capacity(bytes.len());
        let mut vocab = HashMap::new();
        let mut counts = HashMap::new();

        for &b in bytes {
            let id = b as u32;
            sequence.push(id);
            vocab.entry(id).or_insert_with(|| vec![b]);
            *counts.entry(id).or_insert(0) += 1;
        }

        Self { sequence, vocab, counts, next_token_id: 256 }
    }

    fn compute_delta_cic(&self, m: usize, n_x: usize, n_y: usize) -> f64 {
        let m_f = m as f64; let nx_f = n_x as f64; let ny_f = n_y as f64;
        let n_old = self.sequence.len() as f64; let a_old = self.vocab.len() as f64;
        let d_old = a_old - 1.0;
        
        let n_new = n_old - m_f; let a_new = a_old + 1.0; let d_new = a_new - 1.0;
        let nx_new = nx_f - m_f; let ny_new = ny_f - m_f;

        let delta_ll = x_ln_x(nx_f) + x_ln_x(ny_f) - x_ln_x(nx_new) - x_ln_x(ny_new) - x_ln_x(m_f) + x_ln_x(n_new) - x_ln_x(n_old);
        let delta_bic = (d_new / 2.0) * (n_new / (2.0 * PI)).ln() - (d_old / 2.0) * (n_old / (2.0 * PI)).ln();
        let delta_ln_v = 0.5 * PI.ln() - ln_gamma(a_new / 2.0) + ln_gamma(a_old / 2.0);

        let cur_old_x = 1.0 / (24.0 * (nx_f + 0.5)); let cur_old_y = 1.0 / (24.0 * (ny_f + 0.5));
        let cur_new_x = 1.0 / (24.0 * (nx_new + 0.5)); let cur_new_y = 1.0 / (24.0 * (ny_new + 0.5));
        let cur_new_z = 1.0 / (24.0 * (m_f + 0.5));
        
        let local_cur_shift = cur_new_x + cur_new_y + cur_new_z - cur_old_x - cur_old_y;
        let global_cur_shift = (a_new / (12.0 * (n_new + a_new / 2.0))) - (a_old / (12.0 * (n_old + a_old / 2.0)));
        let delta_r = local_cur_shift + global_cur_shift;

        delta_ll + delta_bic + delta_ln_v + delta_r
    }

    pub fn train_fast(&mut self, max_merges: usize) -> PyResult<usize> {
        let mut merges_done = 0;
        for _ in 0..max_merges {
            let mut pair_counts = HashMap::new();
            for window in self.sequence.windows(2) {
                *pair_counts.entry(TokenPair(window[0], window[1])).or_insert(0) += 1;
            }
            if pair_counts.is_empty() { break; }

            let mut best_pair = None;
            let mut best_delta = 0.0; 

            for (pair, &m) in pair_counts.iter() {
                let n_x = *self.counts.get(&pair.0).unwrap();
                let n_y = *self.counts.get(&pair.1).unwrap();
                let delta = self.compute_delta_cic(m, n_x, n_y);

                if delta < best_delta {
                    best_delta = delta;
                    best_pair = Some(*pair);
                }
            }

            if best_delta >= 0.0 || best_pair.is_none() { break; }

            let pair = best_pair.unwrap();
            let mut new_sequence = Vec::with_capacity(self.sequence.len());
            let mut i = 0;
            while i < self.sequence.len() {
                if i < self.sequence.len() - 1 && self.sequence[i] == pair.0 && self.sequence[i+1] == pair.1 {
                    new_sequence.push(self.next_token_id);
                    i += 2;
                } else {
                    new_sequence.push(self.sequence[i]);
                    i += 1;
                }
            }
            self.sequence = new_sequence;
            
            let mut new_token_bytes = self.vocab.get(&pair.0).unwrap().clone();
            new_token_bytes.extend(self.vocab.get(&pair.1).unwrap());
            self.vocab.insert(self.next_token_id, new_token_bytes);
            
            self.counts.clear();
            for &token in &self.sequence {
                *self.counts.entry(token).or_insert(0) += 1;
            }

            self.next_token_id += 1;
            merges_done += 1;
        }
        Ok(merges_done)
    }

    pub fn vocab_size(&self) -> usize { self.vocab.len() }

    #[pyo3(text_signature = "($self, text, /)")]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut tokens = Vec::new();
        let mut i = 0;

        while i < bytes.len() {
            let mut best_match_id = None;
            let mut best_match_len = 0;

            for (&id, token_bytes) in &self.vocab {
                let len = token_bytes.len();
                if len <= bytes.len() - i && &bytes[i..i+len] == token_bytes.as_slice() {
                    if len > best_match_len {
                        best_match_len = len;
                        best_match_id = Some(id);
                    }
                }
            }

            match best_match_id {
                Some(id) => {
                    tokens.push(id);
                    i += best_match_len;
                }
                None => {
                    tokens.push(bytes[i] as u32);
                    i += 1;
                }
            }
        }
        tokens
    }

    #[pyo3(text_signature = "($self, tokens, /)")]
    pub fn decode(&self, tokens: Vec<u32>) -> String {
        let mut bytes = Vec::new();
        for token in tokens {
            if let Some(token_bytes) = self.vocab.get(&token) {
                bytes.extend_from_slice(token_bytes);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

// ==========================================
// 3. PYO3 MODULE BINDING
// ==========================================

#[pymodule]
fn cic_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CICTokenizer>()?;
    Ok(())
}

