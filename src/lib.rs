use pyo3::prelude::*;
use hashbrown::HashMap;
use std::f64::consts::PI;
use statrs::function::gamma::ln_gamma;

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
            let mut best_delta = 0.0; // Must be strictly negative to authorize merge

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

            // Apply merge
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
            
            // Update maps
            let mut new_token_bytes = self.vocab.get(&pair.0).unwrap().clone();
            new_token_bytes.extend(self.vocab.get(&pair.1).unwrap());
            self.vocab.insert(self.next_token_id, new_token_bytes);
            
            // Recompute counts for the next iteration
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

    /// Converts a raw string into an array of optimized CIC token IDs
    #[pyo3(text_signature = "($self, text, /)")]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut tokens = Vec::new();
        let mut i = 0;

        // Greedy longest-prefix match
        while i < bytes.len() {
            let mut best_match_id = bytes[i] as u32; // Fallback to the single atomic byte
            let mut best_match_len = 1;

            for (&id, token_bytes) in &self.vocab {
                let len = token_bytes.len();
                // If this token is longer than our current best, and it fits in the remaining text
                if len > best_match_len && i + len <= bytes.len() {
                    // Check if the byte slice exactly matches
                    if &bytes[i..i+len] == token_bytes.as_slice() {
                        best_match_id = id;
                        best_match_len = len;
                    }
                }
            }
            tokens.push(best_match_id);
            i += best_match_len;
        }

        tokens
    }

    /// Converts an array of CIC token IDs back into a human-readable string
    #[pyo3(text_signature = "($self, tokens, /)")]
    pub fn decode(&self, tokens: Vec<u32>) -> String {
        let mut bytes = Vec::new();
        
        for token in tokens {
            // Retrieve the raw bytes for each token ID, ignoring unrecognized tokens
            if let Some(token_bytes) = self.vocab.get(&token) {
                bytes.extend_from_slice(token_bytes);
            }
        }
        
        // Safely convert the raw bytes back to a string.
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

// The module initialization must stay at the very bottom
#[pymodule]
fn cic_tokenizer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CICTokenizer>()?;
    Ok(())
}

