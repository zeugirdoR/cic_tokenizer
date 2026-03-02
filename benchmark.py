import cic_tokenizer
import tiktoken
import math
from collections import Counter
import urllib.request

def calculate_entropy(encoded_sequence):
    """Calculates the Shannon entropy of the encoded token sequence."""
    counts = Counter(encoded_sequence)
    total_tokens = len(encoded_sequence)
    
    entropy = 0.0
    for count in counts.values():
        p_i = count / total_tokens
        entropy -= p_i * math.log2(p_i)
    return entropy

print("--- CIC vs BPE Benchmark ---")

# 1. Load a real, natural language corpus (e.g., Alice in Wonderland from Project Gutenberg)
url = "https://www.gutenberg.org/files/11/11-0.txt"
print("Downloading natural language corpus...")
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response:
    corpus = response.read().decode('utf-8')[:100000] # Grab the first 100k characters

print(f"Original String Length (Bytes): {len(corpus)}\n")

# ==========================================
# CIC TOKENIZER (GEOMETRIC FLOW)
# ==========================================
print("Training CIC Tokenizer...")
cic = cic_tokenizer.CICTokenizer(corpus)

# Train until the geometry stops authorizing merges (or hits a high cap)
cic_merges = cic.train_fast(5000) 
cic_encoded = cic.encode(corpus)
cic_entropy = calculate_entropy(cic_encoded)

# ==========================================
# BPE TOKENIZER (TIKTOKEN - GPT-4)
# ==========================================
print("Loading BPE Tokenizer (cl100k_base)...")
enc_bpe = tiktoken.get_encoding("cl100k_base")
bpe_encoded = enc_bpe.encode(corpus)
bpe_entropy = calculate_entropy(bpe_encoded)

# ==========================================
# THE RESULTS
# ==========================================
print("\n--- BENCHMARK RESULTS ---")
print(f"CIC Sequence Length: {len(cic_encoded):,} tokens")
print(f"BPE Sequence Length: {len(bpe_encoded):,} tokens")

# A lower ratio means CIC compressed the text into fewer tokens than BPE
compression_ratio = len(cic_encoded) / len(bpe_encoded)
print(f"CIC Compression Advantage: {compression_ratio:.2f}x better than BPE\n")

print(f"CIC Vocabulary Entropy: {cic_entropy:.4f} bits")
print(f"BPE Vocabulary Entropy: {bpe_entropy:.4f} bits")

# Verify Lossless Reconstruction
assert cic.decode(cic_encoded) == corpus, "CRITICAL FAULT: Lossless reconstruction failed!"
