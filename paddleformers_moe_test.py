#!/usr/bin/env python3
"""
PaddleFormers MOE Standalone Benchmark Test.
Direct PaddleFormers API call for MOE models - baseline to compare with FastDeploy fallback.
"""
import time
from paddleformers.transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import paddle
import numpy as np
import os

# ========== 配置 ==========
# MOE 模型路径
MODEL_PATH = os.getenv("MOE_MODEL_PATH", "/path/to/your/moe/model")
# 数据类型 (MOE 模型通常使用 bfloat16)
DTYPE = os.getenv("DTYPE", "bfloat16")  # float16, bfloat16
# Attention 实现
ATTN_IMPL = os.getenv("ATTN_IMPL", "sdpa")  # sdpa, eager, flash_attn
# 最大生成长度
MAX_NEW_TOKENS = 2048
# 是否使用 KV cache
USE_CACHE = True

# ========== 使用说明 ==========
print("""
================================================================================
PaddleFormers MOE Standalone Benchmark Test
================================================================================

使用方法:
1. 设置环境变量指定模型路径:
   export MOE_MODEL_PATH=/path/to/your/moe/model

2. 运行测试:
   python3 paddleformers_moe_test.py

3. 可选参数:
   export DTYPE=bfloat16          # 数据类型 (默认 bfloat16)
   export ATTN_IMPL=sdpa          # Attention 实现 (默认 sdpa)
   export USE_CACHE=true            # 是否使用 KV cache (默认 true)

支持的 MOE 模型:
- Qwen3-MoE (PaddlePaddle/Qwen3-MoE-A14B)
- DeepSeek-MoE (deepseek-ai/DeepSeek-MoE-16b)
- Mixtral-8x7B (mistralai/Mixtral-8x7B)
- 其他基于 PaddleFormers 的 MOE 架构模型

================================================================================
""")

def main(prompt):
    print("=" * 60)
    print("PaddleFormers MOE Standalone Benchmark")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Dtype: {DTYPE}")
    print(f"Attention Impl: {ATTN_IMPL}")
    print(f"Use Cache: {USE_CACHE}")
    print("=" * 60)

    # 检查模型路径
    if MODEL_PATH == "/path/to/your/moe/model":
        print("\n[WARNING] MOE_MODEL_PATH not set!")
        print("Please set: export MOE_MODEL_PATH=/path/to/your/moe/model\n")

    # ========== 显存监控 ==========
    def get_memory_info():
        allocated_mb = paddle.device.cuda.memory_allocated() / 1024**2
        reserved_mb = paddle.device.cuda.memory_reserved() / 1024**2
        return allocated_mb, reserved_mb

    # Check GPU memory before loading
    print("\n=== GPU Memory Before Model Load ===")
    alloc_before, resv_before = get_memory_info()
    print(f"Allocated: {alloc_before:.1f} MB")
    print(f"Reserved:  {resv_before:.1f} MB")

    # ========== 加载模型 ==========
    print("\nLoading tokenizer and config...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        print("Loading config...")
        config = AutoConfig.from_pretrained(
            MODEL_PATH,
            dtype=DTYPE,
            use_cache=USE_CACHE,
            _attn_implementation=ATTN_IMPL,
        )

        # MOE 特定配置
        print(f"\nModel Configuration:")
        print(f"  Architecture: {getattr(config, 'architectures', ['N/A'])[0]}")
        print(f"  Hidden size: {getattr(config, 'hidden_size', 'N/A')}")
        print(f"  Num layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
        print(f"  Num experts: {getattr(config, 'num_experts', getattr(config, 'num_local_experts', 'N/A'))}")
        print(f"  Experts per token: {getattr(config, 'num_experts_per_tok', getattr(config, 'top_k', 'N/A'))}")
        print(f"  Intermediate size: {getattr(config, 'intermediate_size', 'N/A')}")

        print("\nLoading model...")
        print(f">>> dtype: {DTYPE}")
        print(f">>> attention_implementation: {ATTN_IMPL}")
        load_start = time.time()

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=DTYPE,
            config=config,
            convert_from_hf=True,
        ).eval()

        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.2f}s")

    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        print("\nTroubleshooting:")
        print("1. Check if model path is correct")
        print("2. Ensure model is a MOE architecture supported by PaddleFormers")
        print("3. Verify GPU has sufficient memory (MOE models need more VRAM)")
        return

    # Check GPU memory after loading
    print("\n=== GPU Memory After Model Load ===")
    alloc_after_load, resv_after_load = get_memory_info()
    print(f"Allocated: {alloc_after_load:.1f} MB")
    print(f"Reserved:  {resv_after_load:.1f} MB")
    print(f"Delta:     {alloc_after_load - alloc_before:.1f} MB")

    # ========== 模型信息 ==========
    print(f"\nTokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Model vocab size: {config.vocab_size}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")

    # ========== 准备输入 ==========
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{prompt}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pd", add_special_tokens=True)
    input_ids_list = inputs["input_ids"][0].tolist()
    input_len = len(input_ids_list)
    print(f"\nInput token count: {input_len}")

    # ========== Warmup ==========
    print("Warmup (2 runs)...")
    for _ in range(2):
        with paddle.no_grad():
            _ = model.model(inputs["input_ids"], use_cache=USE_CACHE)
        paddle.device.synchronize()

    # Check GPU memory after warmup
    print("\n=== GPU Memory After Warmup ===")
    alloc_after_warmup, _ = get_memory_info()
    print(f"Allocated: {alloc_after_warmup:.1f} MB")

    # ========== Prefill Benchmark (TTFT) ==========
    print("\nPrefill Benchmark (10 runs)...")
    times = []
    for _ in range(10):
        paddle.device.synchronize()
        start = time.time()
        with paddle.no_grad():
            _ = model.model(inputs["input_ids"], use_cache=USE_CACHE)
        paddle.device.synchronize()
        times.append(time.time() - start)

    times = np.array(times)
    ttft_mean = times.mean() * 1000
    ttft_std = times.std() * 1000
    print(f"TTFT Mean: {ttft_mean:.2f}ms ± {ttft_std:.2f}ms")

    # ========== 生成 Benchmark ==========
    print("\nBenchmark run...")
    paddle.device.synchronize()
    start_time = time.time()

    with paddle.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # Greedy Search for deterministic output
        )

    paddle.device.synchronize()
    end_time = time.time()

    # Check GPU memory after generation
    print("\n=== GPU Memory After Generation ===")
    alloc_after_gen, resv_after_gen = get_memory_info()
    print(f"Allocated: {alloc_after_gen:.1f} MB")
    print(f"Reserved:  {resv_after_gen:.1f} MB")
    print(f"Delta:     {alloc_after_gen - alloc_after_load:.1f} MB")

    # ========== 解析输出 ==========
    print(f"\n=== Output Debug ===")
    print(f"outputs type: {type(outputs)}")

    # Handle different output formats
    if isinstance(outputs[0], paddle.Tensor):
        output_ids = outputs[0][0].tolist() if outputs[0].dim() > 1 else outputs[0].tolist()
    else:
        output_ids = outputs[0][0].tolist() if hasattr(outputs[0][0], 'tolist') else list(outputs[0][0])

    output_len = len(output_ids) - input_len  # Subtract input tokens
    elapsed_time = end_time - start_time
    tokens_per_second = output_len / elapsed_time if elapsed_time > 0 else 0

    # ========== 打印结果 ==========
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Input tokens:     {input_len}")
    print(f"Output tokens:    {output_len}")
    print(f"Time elapsed:     {elapsed_time:.2f}s")
    print(f"Throughput:       {tokens_per_second:.2f} tokens/s")
    print(f"TTFT (mean):     {ttft_mean:.2f}ms")
    print(f"Memory (load):    {alloc_after_load:.1f} MB")
    print(f"Memory (peak):    {alloc_after_gen:.1f} MB")
    print("=" * 60)

    # 显示输出
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"\n=== Generated Output ===")
    print(output_text[:500] + "..." if len(output_text) > 500 else output_text)


# ========== 测试 Prompt ==========

# 简单对话测试
simple_prompt = "你好，请介绍一下你自己"

# MOE 模型擅长任务：知识问答
knowledge_prompt = """请解释量子计算的基本原理，以及它与经典计算的区别。"""

# 代码生成任务
code_prompt = """请用Python写一个快速排序算法，并添加详细注释。"""

# 长文本理解任务
long_prompt = """请阅读以下内容并回答问题：

Mixture of Experts (MoE) 是一种神经网络架构，通过将模型分割为多个"专家"子网络和一个路由网络来工作。
对于每个输入，路由网络会选择最合适的专家来处理。这种架构有两个主要优势：
1. 参数效率高：可以增加模型参数量而不线性增加计算量
2. 专业性强：不同专家可以学习处理不同类型的任务

问题：
1. MoE 架构的核心思想是什么？
2. MoE 如何实现参数效率？
3. 请举一个 MoE 在实际应用中的例子。"""

# 推理任务
reasoning_prompt = """如果今天比昨天热，昨天比前天热，而前天的温度是20度。
请推断今天的温度范围，并说明推理过程。"""

# 创意写作
creative_prompt = """请写一首关于人工智能的现代诗，要求：
1. 包含对技术发展的思考
2. 富有诗意和想象
3. 长度不超过100字"""

# English prompt
english_prompt = "Explain the concept of Mixture of Experts in neural networks."

# ========== 主程序 ==========

if __name__ == "__main__":
    # 运行测试 (选择一个 prompt)
    main(simple_prompt)
    # main(knowledge_prompt)
    # main(code_prompt)
    # main(long_prompt)
    # main(english_prompt)
