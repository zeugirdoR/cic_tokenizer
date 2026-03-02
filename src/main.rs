use std::collections::HashMap;

// Assuming your Cargo.toml package name is `cic_tokenizer`
use cic_tokenizer::geometry::TokenizerNode;
use cic_tokenizer::stream_text_to_geometry; 

fn main() {
    println!("--- Initializing CIC Geometric Tokenizer ---");

    // Initialize the spacetime manifold for our token network
    let mut network: HashMap<String, TokenizerNode> = HashMap::new();
    
    // The coupling constant (learning rate for the Ricci flow)
    let kappa = 0.05; 

    // A tiny synthetic corpus to test the plumbing
    let corpus = "the quick brown fox jumps over the lazy dog";

    println!("Streaming corpus to geometry engine...");
    
    // Fire the stream (this calls the function from lib.rs)
    stream_text_to_geometry(corpus, &mut network, kappa);

    println!("Manifold updated. Tracking {} unique topological edges.", network.len());
    
    // Print a sample edge to verify the geometry shifted
    if let Some(node) = network.get("th") {
        println!("Edge 't' -> 'h': Theta = [{:.4}, {:.4}, {:.4}], Mass = {}", 
            node.theta[0], node.theta[1], node.theta[2], node.n_samples);
    } else {
        println!("Edge 'th' not found in this corpus.");
    }
}
