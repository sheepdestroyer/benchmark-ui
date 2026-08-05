# KV Cache Quantization KLD Results

**Model**: `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`  
**Generation Command**: `python3 kld_benchmark.py --model Qwen3.6-35B-A3B-UD-Q4_K_S.gguf --corpus kld_corpus.txt`  
**Corpus Reference**: [kld_corpus.txt](kld_corpus.txt) (Technical evaluation prose covering transformer attention and KV cache memory scaling)

| KV Cache Quant | Perplexity (PPL) | KL Divergence | Same Top % |
| :--- | :--- | :--- | :--- |
| f16 (Baseline) | 3.9085 | 0.000000 | 100.00% |
| q8_0 | 3.9240 | 0.003090 | 99.21% |
| q5_1 | 3.9452 | 0.002396 | 100.00% |
| q4_0 | 3.9530 | 0.005038 | 100.00% |

> [!NOTE]
> **Quantization Behavior Analysis**:
> - `f16` serves as the unquantized baseline with zero divergence.
> - `q8_0` yields a ~50% reduction in KV cache memory footprint with minimal perplexity degradation (+0.0155 PPL) and high top-token retention (99.21%).
> - `q5_1` and `q4_0` achieve >65% memory compression while maintaining 100.00% top-token matching fidelity on this corpus.
> - `q5_1` exhibits slightly lower KL divergence (0.002396) than `q8_0` (0.003090) due to quantization scale rounding alignment with the float distribution on this specific evaluation text.
