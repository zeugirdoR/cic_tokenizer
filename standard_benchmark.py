import cic_tokenizer
import tiktoken
import math
import urllib.request
from collections import Counter

def get_corpus(chars=100000):
    url = "https://www.gutenberg.org/files/11/11-0.txt" # Alice in Wonderland
    req = urllib.request.Request(url, headers={'User-Agent': 'Gemini-CIC-Bench'})
    with urllib.request.urlopen(req) as res:
        return res.read().decode('utf-8')[:chars]

def calculate_stats(encoded):
    counts = Counter(encoded)
    total = len(encoded)
    entropy = -sum((c/total) * math.log2(c/total) for c in counts.values())
    return total, entropy

# --- EXECUTION ---
text = get_corpus()
print(f"Corpus Size: {len(text)} characters")

# 1. CIC (Geometric Flow)
cic = cic_tokenizer.CICTokenizer(text)
cic.train_fast(2500) # Target crossing point observed in Pareto frontier
cic_len, cic_entropy = calculate_stats(cic.encode(text))

# 2. BPE (GPT-4)
bpe_enc = tiktoken.get_encoding("cl100k_base")
bpe_len, bpe_entropy = calculate_stats(bpe_enc.encode(text))

# --- REPORT ---
ratio = cic_len / bpe_len
print(f"\n{'='*30}\nFINAL BENCHMARK RESULTS\n{'='*30}")
print(f"CIC Sequence Length: {cic_len:,} tokens")
print(f"BPE Sequence Length: {bpe_len:,} tokens")
print(f"Compression Ratio:   {ratio:.4f}x (Target: 0.84x)")
print(f"CIC Entropy:         {cic_entropy:.4f} bits")
