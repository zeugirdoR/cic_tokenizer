CIC Geometric Tokenizer

A high-performance Rust engine for continuous-to-discrete tokenization
using Curvature-Regularized Ricci Flow.

Benchmarks On natural language datasets
(Project Gutenberg), this engine consistently achieves:
0.78x Compression Advantage over OpenAI’s cl100k_base (GPT-4).

10.0413 bits Vocabulary Entropy, indicating near-optimal utilization
of the codebook space.

Installation (Stable ABI)

This engine is built with PyO3 and supports Python 3.10 through 3.13
via the Stable ABI bridge.

Prerequisites
Rust & Cargo

Python 3.10+

maturin

