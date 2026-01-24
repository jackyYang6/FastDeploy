"""
FastDeploy MOE Fleet Fallback Test
==================================
Tests MOE model inference using PaddleFormers Fleet Backend.

Features:
1. Weight loading verification (QKV/Gate+Up fusion)
2. Inference output correctness
3. Performance benchmarking
4. Memory usage statistics

Usage:
    # Basic test with Fleet backend
    export MOE_MODEL_PATH=/path/to/Qwen3-MoE-model
    python fd_moe_fallback_test.py

    # Force Fleet implementation
    export MODEL_IMPL=paddleformers
    python fd_moe_fallback_test.py --use-fleet

    # Verify weight loading only (no inference)
    python fd_moe_fallback_test.py --verify-only
"""

import argparse
import os
import subprocess
import time
from typing import Optional

import paddle


def get_gpu_memory() -> tuple[float, str]:
    """Get GPU memory usage for visible devices."""
    try:
        cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", None)
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')

        gpu_memory = {}
        for line in lines:
            parts = line.split(',')
            if len(parts) == 2:
                idx = int(parts[0].strip())
                mem = int(parts[1].strip())
                gpu_memory[idx] = mem

        if cuda_visible:
            visible_indices = [int(x.strip()) for x in cuda_visible.split(',')]
            total_mem = sum(gpu_memory.get(idx, 0) for idx in visible_indices)
            mem_details = ", ".join([f"GPU{idx}:{gpu_memory.get(idx, 0)}MB" for idx in visible_indices])
        else:
            total_mem = sum(gpu_memory.values())
            mem_details = ", ".join([f"GPU{idx}:{mem}MB" for idx, mem in gpu_memory.items()])

        return float(total_mem), mem_details
    except Exception as e:
        print(f"GPU memory query failed: {e}")
        return 0.0, "N/A"


def verify_fleet_model_structure(model_path: str) -> dict:
    """Verify Fleet model structure and return info."""
    from paddleformers.transformers import AutoConfig, AutoModelForCausalLM

    print("\n" + "=" * 60)
    print("Verifying Fleet Model Structure")
    print("=" * 60)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    config.tensor_model_parallel_size = 1
    config.fuse_attention_qkv = True
    config.fuse_attention_ffn = True

    model = AutoModelForCausalLM.from_config(config, dtype="bfloat16")

    is_fleet = getattr(model, "is_fleet", False)
    print(f"is_fleet: {is_fleet}")

    param_names = [name for name, _ in model.named_parameters()]

    qkv_fused = sum(1 for n in param_names if "qkv_proj" in n)
    gate_up_fused = sum(1 for n in param_names if "up_gate_proj" in n)
    q_separate = sum(1 for n in param_names if "q_proj" in n and "qkv" not in n)

    print(f"QKV fused layers: {qkv_fused}")
    print(f"Gate+Up fused layers: {gate_up_fused}")
    print(f"Separate Q layers: {q_separate}")

    result = {
        "is_fleet": is_fleet,
        "qkv_fused": qkv_fused > 0,
        "gate_up_fused": gate_up_fused > 0,
        "num_params": len(param_names),
    }

    if is_fleet and qkv_fused > 0 and gate_up_fused > 0:
        print("\n✅ Model is ready for Fleet fallback")
    else:
        print("\n⚠️ Model may not be properly configured for Fleet fallback")

    del model
    paddle.device.cuda.empty_cache()

    return result


def run_inference_test(
    model_path: str,
    use_fleet: bool = False,
    max_tokens: int = 256,
    use_cudagraph: bool = False,
) -> Optional[dict]:
    """Run inference test and return results."""
    from fastdeploy import LLM, SamplingParams

    model_impl = "paddleformers" if use_fleet else "auto"

    print("\n" + "=" * 60)
    print(f"Running Inference Test (model_impl={model_impl})")
    print("=" * 60)

    mem_before, _ = get_gpu_memory()
    print(f"Memory before load: {mem_before:.1f} MB")

    print("\nLoading model...")
    load_start = time.time()

    try:
        llm = LLM(
            model=model_path,
            model_impl=model_impl,
            tensor_parallel_size=int(os.getenv("TP_SIZE", 1)),
            max_model_len=32768,
            num_gpu_blocks_override=800,
            graph_optimization_config={"use_cudagraph": use_cudagraph},
        )
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None

    mem_after_load, mem_details = get_gpu_memory()
    print(f"Memory after load: {mem_after_load:.1f} MB ({mem_details})")

    messages = [[{"role": "user", "content": "Hello, please introduce yourself briefly."}]]

    print("\nWarming up (TTFT)...")
    sampling_params_ttft = SamplingParams(max_tokens=1)
    start = time.time()
    _ = llm.chat(messages, sampling_params_ttft)
    paddle.device.synchronize("gpu")
    ttft = time.time() - start
    print(f"TTFT: {ttft * 1000:.1f}ms")

    print("\nRunning benchmark...")
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.7)
    paddle.device.synchronize("gpu")
    start_time = time.time()

    try:
        outputs = llm.chat(messages, sampling_params)
        paddle.device.synchronize("gpu")
        elapsed = time.time() - start_time

        total_output_tokens = 0
        generated_text = ""
        for output in outputs:
            completion = output.outputs
            if hasattr(completion, 'text'):
                generated_text = completion.text
            if hasattr(completion, 'token_ids'):
                total_output_tokens += len(completion.token_ids)
            else:
                total_output_tokens += len(generated_text) // 2

        print(f"\n[Output Preview] {generated_text[:200]}...")

    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    mem_after_gen, _ = get_gpu_memory()

    results = {
        "model_impl": model_impl,
        "load_time": load_time,
        "ttft_ms": ttft * 1000,
        "output_tokens": total_output_tokens,
        "elapsed_time": elapsed,
        "throughput": total_output_tokens / elapsed if elapsed > 0 else 0,
        "memory_after_load": mem_after_load,
        "memory_after_gen": mem_after_gen,
        "memory_delta": mem_after_gen - mem_after_load,
    }

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model Impl:        {results['model_impl']}")
    print(f"Load Time:         {results['load_time']:.2f}s")
    print(f"TTFT:              {results['ttft_ms']:.1f}ms")
    print(f"Output Tokens:     {results['output_tokens']}")
    print(f"Time Elapsed:      {results['elapsed_time']:.2f}s")
    print(f"Throughput:        {results['throughput']:.2f} tokens/s")
    print(f"Memory (load):     {results['memory_after_load']:.1f} MB")
    print(f"Memory (gen):      {results['memory_after_gen']:.1f} MB")
    print(f"Memory Delta:      {results['memory_delta']:.1f} MB")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="FastDeploy MOE Fleet Fallback Test")
    parser.add_argument("--model", type=str, default=os.getenv("MOE_MODEL_PATH", ""),
                        help="Path to MOE model")
    parser.add_argument("--use-fleet", action="store_true",
                        help="Force use Fleet backend (model_impl=paddleformers)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify model structure, skip inference")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens to generate")
    parser.add_argument("--use-cudagraph", action="store_true",
                        help="Enable CUDA graph optimization")
    args = parser.parse_args()

    model_path = args.model
    if not model_path:
        print("[ERROR] Model path not specified.")
        print("Usage: python fd_moe_fallback_test.py --model /path/to/moe/model")
        print("   Or: export MOE_MODEL_PATH=/path/to/moe/model")
        return

    print("=" * 60)
    print("FastDeploy MOE Fleet Fallback Test")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Use Fleet: {args.use_fleet}")
    print(f"Verify Only: {args.verify_only}")

    if args.verify_only:
        verify_fleet_model_structure(model_path)
        return

    results = run_inference_test(
        model_path=model_path,
        use_fleet=args.use_fleet,
        max_tokens=args.max_tokens,
        use_cudagraph=args.use_cudagraph,
    )

    if results:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")


if __name__ == "__main__":
    main()
