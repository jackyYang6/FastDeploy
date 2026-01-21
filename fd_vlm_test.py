"""
FastDeploy VLM Direct Test - 用于测试 VLM Fallback
绕过 API Server，直接调用 LLM.generate()
"""
import time
import paddle
from fastdeploy import LLM, SamplingParams
import os

# ========== 配置 ==========
MODEL_PATH = "/home/disk2/yangjiajie/Qwen/Qwen3-VL-2B-Instruct"
MODEL_IMPL = os.getenv("MODEL_IMPL", "paddleformers")  # 测试 VLM fallback
MAX_NEW_TOKENS = 512
NUM_GPU_BLOCKS = 400


def get_gpu_memory():
    """获取当前可见 GPU 的总显存占用"""
    import subprocess
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
        else:
            total_mem = sum(gpu_memory.values())
        
        return total_mem
    except Exception as e:
        print(f"GPU memory query failed: {e}")
        return 0


def main_text_only():
    """先测试纯文本模式，不带图片"""
    
    # 纯文本 prompt
    messages = [
        [{"role": "user", "content": "你好，请介绍一下你自己。"}],
    ]
    
    print("=" * 60)
    print(f"FastDeploy VLM Fallback Test (Text-Only)")
    print(f"Model: {MODEL_PATH}")
    print(f"Model Impl: {MODEL_IMPL}")
    print("=" * 60)
    
    alloc_before = get_gpu_memory()
    print(f"\n[Memory] Before load: {alloc_before} MB")
    
    # 加载模型
    print("\nLoading VLM model...")
    load_start = time.time()
    
    llm_kwargs = {
        "model": MODEL_PATH,
        "model_impl": MODEL_IMPL,
        "tensor_parallel_size": int(os.getenv("TP_SIZE", 1)),
        "max_model_len": 4096,
        "num_gpu_blocks_override": NUM_GPU_BLOCKS,
        "graph_optimization_config": {"use_cudagraph": False},  # VLM 先禁用 CUDA Graph
    }
    
    llm = LLM(**llm_kwargs)
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")
    
    alloc_after_load = get_gpu_memory()
    print(f"[Memory] After load: {alloc_after_load} MB")
    
    # 采样参数
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0,
    )
    
    # 生成
    print("\nGenerating...")
    paddle.device.synchronize("gpu")
    start_time = time.time()
    
    outputs = llm.chat(messages, sampling_params)
    
    paddle.device.synchronize("gpu")
    elapsed = time.time() - start_time
    
    # 统计结果
    for output in outputs:
        completion = output.outputs
        generated_text = completion.text if hasattr(completion, 'text') else str(completion)
        
        if hasattr(completion, 'token_ids'):
            total_output_tokens = len(completion.token_ids)
        else:
            total_output_tokens = len(generated_text) // 2
            
        print(f"\n[Output] {generated_text[:500]}...")
    
    # 打印结果
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model Impl:        {MODEL_IMPL}")
    print(f"Output tokens:     {total_output_tokens}")
    print(f"Time elapsed:      {elapsed:.2f}s")
    print(f"Throughput:        {total_output_tokens / elapsed:.2f} tokens/s")
    print("=" * 60)


def main_with_image():
    """测试带图片的多模态推理"""
    from PIL import Image
    import numpy as np
    
    # 创建一个测试图片 (可以替换成真实图片路径)
    test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # test_image = Image.open("/path/to/test_image.jpg")
    
    # 多模态 prompt with image
    messages = [
        [{
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": "请描述这张图片"},
            ]
        }],
    ]
    
    print("=" * 60)
    print(f"FastDeploy VLM Fallback Test (With Image)")
    print(f"Model: {MODEL_PATH}")
    print(f"Model Impl: {MODEL_IMPL}")
    print("=" * 60)
    
    # 加载模型
    print("\nLoading VLM model...")
    load_start = time.time()
    
    llm_kwargs = {
        "model": MODEL_PATH,
        "model_impl": MODEL_IMPL,
        "tensor_parallel_size": int(os.getenv("TP_SIZE", 1)),
        "max_model_len": 4096,
        "num_gpu_blocks_override": NUM_GPU_BLOCKS,
        "graph_optimization_config": {"use_cudagraph": False},
    }
    
    llm = LLM(**llm_kwargs)
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")
    
    # 采样参数
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0,
    )
    
    # 生成
    print("\nGenerating with image...")
    paddle.device.synchronize("gpu")
    start_time = time.time()
    
    outputs = llm.chat(messages, sampling_params)
    
    paddle.device.synchronize("gpu")
    elapsed = time.time() - start_time
    
    # 统计结果
    for output in outputs:
        completion = output.outputs
        generated_text = completion.text if hasattr(completion, 'text') else str(completion)
        print(f"\n[Output] {generated_text[:500]}...")
    
    print("\n" + "=" * 60)
    print(f"VLM Generation completed in {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    # 先测试纯文本模式
    main_text_only()
    
    # 如果纯文本通过，再测试带图片的
    # main_with_image()
