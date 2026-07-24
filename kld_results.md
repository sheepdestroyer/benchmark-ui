# KV Cache Quantization KLD Results

**Model**: `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`

| KV Cache Quant | Perplexity (PPL) | KL Divergence | Same Top % |
| :--- | :--- | :--- | :--- |
| f16 (Baseline) | 3.9085 | 0.000000 | 100.00% |
| q8_0 | 3.9240 | 0.003090 | 99.21% |
| q5_1 | 3.9452 | 0.002396 | 100.00% |
| q4_0 | 3.9530 | 0.005038 | 100.00% |
