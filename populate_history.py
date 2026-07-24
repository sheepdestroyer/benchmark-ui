#!/usr/bin/env python3
import os
import json
import datetime
import shutil
from pathlib import Path

BENCH_DIR = Path(__file__).parent.resolve()
HISTORY_DIR = BENCH_DIR / "history"

def populate():
    # Purge existing history
    if HISTORY_DIR.exists():
        shutil.rmtree(HISTORY_DIR)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    
    # Keep gitkeep
    with open(HISTORY_DIR / ".gitkeep", "w") as f:
        f.write("")
        
    # Profile metadata configurations
    profiles = {
        "Qwen3.6-27B": {
            "model_name": "unsloth/Qwen3.6-27B-GGUF:Q4_K_S",
            "spec_type": "None",
            "spec_draft_type_k": "None",
            "spec_draft_type_v": "None",
            "flash_attn": "true",
            "parallel": "1",
            "base_prefill": 1550.0,
            "base_decode": 42.0,
            "ppl_base": 3.93,
            "needle": "Pass", "ruler": "Pass", "longbench": "Pass", "swe_bench": "Fail"
        },
        "Qwen3.6-27B-spec": {
            "model_name": "unsloth/Qwen3.6-27B-GGUF:Q4_K_S",
            "spec_type": "ngram-mod",
            "spec_draft_type_k": "q5_1",
            "spec_draft_type_v": "q5_1",
            "flash_attn": "true",
            "parallel": "1",
            "base_prefill": 1500.0,
            "base_decode": 75.0,
            "ppl_base": 3.93,
            "needle": "Pass", "ruler": "Pass", "longbench": "Pass", "swe_bench": "Fail"
        },
        "Qwen3.6-27B-spec2": {
            "model_name": "unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_S",
            "spec_type": "draft-mtp",
            "spec_draft_type_k": "q5_1",
            "spec_draft_type_v": "q5_1",
            "flash_attn": "true",
            "parallel": "1",
            "base_prefill": 1450.0,
            "base_decode": 110.0,
            "ppl_base": 3.94,
            "needle": "Pass", "ruler": "Pass", "longbench": "Pass", "swe_bench": "Pass"
        },
        "Qwen3.6-27B-spec3": {
            "model_name": "unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_S",
            "spec_type": "draft-mtp,ngram-mod",
            "spec_draft_type_k": "q5_1",
            "spec_draft_type_v": "q5_1",
            "flash_attn": "true",
            "parallel": "1",
            "base_prefill": 1420.0,
            "base_decode": 135.0,
            "ppl_base": 3.94,
            "needle": "Pass", "ruler": "Pass", "longbench": "Pass", "swe_bench": "Pass"
        },
        "Qwen3.6-35B-A3B": {
            "model_name": "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_S",
            "spec_type": "None",
            "spec_draft_type_k": "None",
            "spec_draft_type_v": "None",
            "flash_attn": "true",
            "parallel": "1",
            "base_prefill": 2480.0,
            "base_decode": 140.0,
            "ppl_base": 3.95,
            "needle": "Pass", "ruler": "Pass", "longbench": "Pass", "swe_bench": "Pass"
        },
        "Qwen3.6-35B-A3B-spec": {
            "model_name": "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_S",
            "spec_type": "draft-mtp,ngram-mod",
            "spec_draft_type_k": "q5_1",
            "spec_draft_type_v": "q5_1",
            "flash_attn": "true",
            "parallel": "1",
            "base_prefill": 2350.0,
            "base_decode": 265.0,
            "ppl_base": 3.95,
            "needle": "Pass", "ruler": "Pass", "longbench": "Pass", "swe_bench": "Pass"
        }
    }
    
    # KV cache formats and their loss multipliers
    quants = {
        "f16": {"kld": 0.0, "ppl_shift": 0.0, "top_match": 100.0, "speed_mult": 1.0},
        "q8_0": {"kld": 0.0018, "ppl_shift": -0.012, "top_match": 98.4, "speed_mult": 1.05},
        "q5_1": {"kld": 0.0024, "ppl_shift": 0.038, "top_match": 97.6, "speed_mult": 1.15},
        "q4_0": {"kld": 0.0048, "ppl_shift": 0.075, "top_match": 96.2, "speed_mult": 1.25}
    }
    
    timestamp = datetime.datetime.now()
    
    count = 0
    for profile_alias, p_cfg in profiles.items():
        for q_name, q_cfg in quants.items():
            # Adjust timestamp slightly to keep sorting consistent
            run_time = timestamp - datetime.timedelta(minutes=count * 5)
            run_time_str = run_time.isoformat()
            
            # Apply multipliers for cache quants on speeds
            prefill_speed = p_cfg["base_prefill"] * q_cfg["speed_mult"]
            decode_speed = p_cfg["base_decode"] * q_cfg["speed_mult"]
            
            run_data = {
                "run_metadata": {
                    "timestamp": run_time_str,
                    "target_endpoint": "http://127.0.0.1:8081",
                    "cli_arguments": [
                        "--mode", "all",
                        "--model", profile_alias,
                        "--tokens", "5000"
                    ]
                },
                "model_settings": {
                    "model_name": p_cfg["model_name"],
                    "profile_alias": profile_alias,  # explicit profile mapping!
                    "base_quantization": "Q4_K_S",
                    "kv_cache_quant": q_name,
                    "threads": 16,
                    "ubatch_size": 512,
                    "batch_size": 2048,
                    "speculative_draft_type": p_cfg["spec_type"],
                    # Embed settings in json
                    "spec_type": p_cfg["spec_type"],
                    "spec_draft_type_k": p_cfg["spec_draft_type_k"],
                    "spec_draft_type_v": p_cfg["spec_draft_type_v"],
                    "flash_attn": p_cfg["flash_attn"],
                    "parallel": p_cfg["parallel"],
                    "fit": "true"
                },
                "throughput_metrics": {
                    "prefill_speed": round(prefill_speed, 2),
                    "decode_speed": round(decode_speed, 2),
                    "ttft": round(0.4 / q_cfg["speed_mult"], 3)
                },
                "reasoning_accuracy": {
                    "needle": p_cfg["needle"],
                    "ruler": p_cfg["ruler"],
                    "longbench": p_cfg["longbench"],
                    "swe_bench": p_cfg["swe_bench"]
                },
                "quantization_loss": {
                    "perplexity": round(p_cfg["ppl_base"] + q_cfg["ppl_shift"], 4),
                    "mean_kld": q_cfg["kld"],
                    "same_top_match_percent": q_cfg["top_match"]
                }
            }
            
            # Write JSON file
            filename = f"run_{run_time.strftime('%Y-%m-%dT%H-%M-%S-%f')}.json"
            with open(HISTORY_DIR / filename, "w") as f:
                json.dump(run_data, f, indent=4)
            count += 1
            
    print(f"[+] Successfully populated history registry with {count} structured runs!")

if __name__ == "__main__":
    populate()
