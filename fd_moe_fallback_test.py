"""
FastDeploy MOE Fallback Direct Test - 用于测试 MOE 模型的 PaddleFormers Fallback
绕过 API Server，直接调用 LLM.generate()
"""
import time
import paddle
from fastdeploy import LLM, SamplingParams
import os

# ========== 配置 ==========
# MOE 模型路径示例 (Qwen3-MoE, Mixtral, DeepSeek-MoE 等)
MODEL_PATH = os.getenv("MOE_MODEL_PATH", "/path/to/your/moe/model")
# 可选: paddleformers 模式会强制使用 fallback
MODEL_IMPL = os.getenv("MODEL_IMPL", "auto")  # auto, paddleformers
# MOE 模型通常显存需求更大，适当调整
MAX_NEW_TOKENS = 2048
NUM_GPU_BLOCKS = 800  # 固定 KV cache blocks 数量，保证显存可比

# ========== MOE 模型配置提示 ==========
print("""
================================================================================
FastDeploy MOE Fallback Test Script
================================================================================

使用方法:
1. 设置环境变量指定模型路径:
   export MOE_MODEL_PATH=/path/to/your/moe/model

2. 运行测试:
   python3 fd_moe_fallback_test.py

3. 可选参数:
   export MODEL_IMPL=paddleformers  # 强制使用 PaddleFormers fallback
   export TP_SIZE=1                # Tensor Parallel size (默认1)

支持的 MOE 模型:
- Qwen3-MoE (PaddlePaddle/Qwen3-MoE-A14B)
- DeepSeek-MoE (deepseek-ai/DeepSeek-MoE-16b)
- Mixtral-8x7B (mistralai/Mixtral-8x7B)
- 其他基于 PaddleFormers 的 MOE 架构模型

================================================================================
""")

def get_gpu_memory():
    """获取当前可见 GPU 的总显存占用 (支持 CUDA_VISIBLE_DEVICES)"""
    import subprocess
    try:
        # 获取 CUDA_VISIBLE_DEVICES 指定的 GPU 索引
        cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", None)

        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')

        # 解析每个 GPU 的显存
        gpu_memory = {}
        for line in lines:
            parts = line.split(',')
            if len(parts) == 2:
                idx = int(parts[0].strip())
                mem = int(parts[1].strip())
                gpu_memory[idx] = mem

        # 根据 CUDA_VISIBLE_DEVICES 过滤
        if cuda_visible:
            visible_indices = [int(x.strip()) for x in cuda_visible.split(',')]
            total_mem = sum(gpu_memory.get(idx, 0) for idx in visible_indices)
            mem_details = ", ".join([f"GPU{idx}:{gpu_memory.get(idx, 0)}MB" for idx in visible_indices])
            print(f"  [GPUs: {mem_details}]")
        else:
            total_mem = sum(gpu_memory.values())

        return total_mem, total_mem
    except Exception as e:
        print(f"GPU memory query failed: {e}")
        return 0, 0

def main(prompt, use_cudagraph=False):
    # ========== 测试 Messages ==========
    messages = [
        [
            {"role": "user", "content": f"{prompt}"},
        ],
    ]
    print("=" * 60)
    print(f"FastDeploy MOE Fallback Test")
    print(f"Model: {MODEL_PATH}")
    print(f"Model Impl: {MODEL_IMPL}")
    print("=" * 60)

    # 初始显存
    alloc_before, _ = get_gpu_memory()
    print(f"\n[Memory] Before load: {alloc_before:.1f} MB")

    # 加载模型
    print("\nLoading model (this may take a while for MOE models)...")
    load_start = time.time()

    # MOE 模型配置
    llm_kwargs = {
        "model": MODEL_PATH,
        "model_impl": MODEL_IMPL,
        "tensor_parallel_size": int(os.getenv("TP_SIZE", 1)),
        "max_model_len": 32768,
        "num_gpu_blocks_override": NUM_GPU_BLOCKS,
    }
    llm_kwargs["graph_optimization_config"] = {"use_cudagraph": use_cudagraph}

    try:
        llm = LLM(**llm_kwargs)
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        print("\nTroubleshooting:")
        print("1. Check if the model path is correct")
        print("2. Ensure the model is a MOE architecture supported by PaddleFormers")
        print("3. Verify GPU has sufficient memory (MOE models need more VRAM)")
        print("4. Try MODEL_IMPL=paddleformers to force fallback")
        return

    alloc_after_load, _ = get_gpu_memory()
    print(f"[Memory] After load: {alloc_after_load:.1f} MB")

    sampling_params_ttft = SamplingParams(max_tokens=1)  # 只生成1个token
    start = time.time()
    _ = llm.chat(messages, sampling_params_ttft)
    paddle.device.synchronize("gpu")
    ttft = time.time() - start
    print(f"TTFT (approx): {ttft * 1000:.1f}ms")

    # 采样参数
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
    )

    # 正式测试
    print("\nBenchmark run...")
    paddle.device.synchronize("gpu")
    start_time = time.time()

    try:
        outputs = llm.chat(messages, sampling_params)

        paddle.device.synchronize("gpu")
        elapsed = time.time() - start_time

        # 统计结果
        total_output_tokens = 0
        for output in outputs:
            # CompletionOutput 直接访问 .text 和 .token_ids
            completion = output.outputs
            generated_text = completion.text if hasattr(completion, 'text') else str(completion)

            # 获取 token 数
            if hasattr(completion, 'token_ids'):
                total_output_tokens += len(completion.token_ids)
            else:
                total_output_tokens += len(generated_text) // 2  # 估算

            print(f"\n[Output] {generated_text}...")
    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}")
        return

    alloc_after_gen, _ = get_gpu_memory()

    # 打印结果
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model Impl:        {MODEL_IMPL}")
    print(f"Output tokens:     {total_output_tokens}")
    print(f"Time elapsed:      {elapsed:.2f}s")
    print(f"Throughput:        {total_output_tokens / elapsed:.2f} tokens/s")
    print(f"Memory after load: {alloc_after_load:.1f} MB")
    print(f"Memory after gen:  {alloc_after_gen:.1f} MB")
    print(f"Memory delta:      {alloc_after_gen - alloc_after_load:.1f} MB")
    print("=" * 60)


# ========== 测试 Prompt ==========

# 简单对话测试
simple_prompt = "你好，请介绍一下你自己"

# 代码生成任务
code_prompt = """请用Python写一个快速排序算法，并添加详细注释。"""

# MOE 模型擅长任务：知识问答
knowledge_prompt = """请解释量子计算的基本原理，以及它与经典计算的区别。"""

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
    # 检查模型路径
    if MODEL_PATH == "/path/to/your/moe/model":
        print("[WARNING] MOE_MODEL_PATH not set, using default placeholder.")
        print("Please set: export MOE_MODEL_PATH=/path/to/your/moe/model\n")

    # 运行测试 (选择一个 prompt)
    main(simple_prompt, use_cudagraph=False)
    # main(knowledge_prompt, use_cudagraph=False)
    # main(code_prompt, use_cudagraph=False)
