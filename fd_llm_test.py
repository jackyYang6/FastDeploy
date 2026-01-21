"""
FastDeploy LLM Direct Test - 用于对比 FD Native / FD Fallback / PaddleFormers
绕过 API Server，直接调用 LLM.generate()
"""
import time
import paddle
from fastdeploy import LLM, SamplingParams
import os

# ========== 配置 ==========
# MODEL_PATH = "/home/disk2/yangjiajie/unsloth/gemma-3-1b-it"
# MODEL_PATH = "/home/disk2/yangjiajie/microsoft/Phi-3-mini-4k-instruct"
MODEL_PATH = "/home/disk2/yangjiajie/Qwen/Qwen3-4B"
# MODEL_PATH = "/home/disk2/yangjiajie/unsloth/Llama-3.1-8B-Instruct"
# MODEL_PATH = "/home/disk2/yangjiajie/unsloth/Llama-3.2-1B-Instruct"
# MODEL_PATH = "/home/disk2/yangjiajie/PaddlePaddle/ERNIE-4.5-0.3B-PT"
MODEL_IMPL = os.getenv("MODEL_IMPL", "auto")  # 测试 fallback with TP replacement
MAX_NEW_TOKENS = 4096
NUM_GPU_BLOCKS = 600  # 固定 KV cache blocks 数量，保证显存可比

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

def main(prompt, use_cudagraph):
    # ========== 测试 Messages ==========
    messages = [
        [
            {"role": "user", "content": f"{prompt}"},
        ],
    ]
    print("=" * 60)
    print(f"FastDeploy LLM Direct Test")
    print(f"Model: {MODEL_PATH}")
    print(f"Model Impl: {MODEL_IMPL}")
    print("=" * 60)
    
    # 初始显存
    alloc_before, _ = get_gpu_memory()
    print(f"\n[Memory] Before load: {alloc_before:.1f} MB")
    
    # 加载模型
    print("\nLoading model...")
    load_start = time.time()
    
    # 对 PaddleFormers Fallback 禁用 CUDA Graph（避免显存翻倍）
    llm_kwargs = {
        "model": MODEL_PATH,
        "model_impl": MODEL_IMPL,
        "tensor_parallel_size": int(os.getenv("TP_SIZE", 1)),
        "max_model_len": 32768,
        "num_gpu_blocks_override": NUM_GPU_BLOCKS,
    }
    llm_kwargs["graph_optimization_config"] = {"use_cudagraph": use_cudagraph}
    
    llm = LLM(**llm_kwargs)
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")
    
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
        temperature=0,
    )
    
    # 正式测试
    print("\nBenchmark run...")
    paddle.device.synchronize("gpu")
    start_time = time.time()
    
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
            print(f"Token IDs: {completion.token_ids}")
        else:
            total_output_tokens += len(generated_text) // 2  # 估算
            print(f"Generated Text: {generated_text}")  
        print(f"\n[Output] {generated_text}...")
    
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


classical_prompt = "请把李白的静夜思改写成现代诗歌"

# 长文本摘要任务 (~800 tokens input)
long_prompt_1 = """请阅读以下文章并给出详细总结：
人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，致力于开发能够执行通常需要人类智能才能完成的任务的系统。这些任务包括学习、推理、问题解决、感知和语言理解。
AI的历史可以追溯到20世纪50年代。1956年，达特茅斯会议被认为是人工智能作为一个学术领域的诞生。早期的AI研究主要集中在符号推理和问题解决上。研究人员开发了能够证明定理和下棋的程序，这在当时被认为是需要智能才能完成的任务。
机器学习是AI的一个子领域，专注于开发能够从数据中学习的算法。传统的机器学习方法包括决策树、支持向量机和朴素贝叶斯分类器。这些方法在许多应用中取得了成功，如垃圾邮件过滤和信用评分。
深度学习是机器学习的一个子集，使用多层神经网络来学习数据的分层表示。深度学习在图像识别、语音识别和自然语言处理等领域取得了突破性进展。卷积神经网络（CNN）特别擅长处理图像数据，而循环神经网络（RNN）和Transformer架构则在序列数据处理方面表现出色。
大型语言模型（LLM）是近年来AI领域最令人兴奋的发展之一。这些模型，如GPT、BERT和LLaMA，通过在海量文本数据上进行预训练，能够执行各种自然语言处理任务。它们可以生成文本、回答问题、翻译语言，甚至编写代码。
AI在各行各业都有广泛的应用。在医疗保健领域，AI被用于疾病诊断、药物发现和个性化治疗。在金融领域，AI被用于欺诈检测、风险评估和算法交易。在制造业，AI驱动的机器人和预测性维护系统提高了生产效率。
尽管AI取得了巨大进步，但仍面临许多挑战。这些挑战包括确保AI系统的公平性和透明度、保护用户隐私、以及解决AI可能带来的就业影响。负责任的AI开发和部署是当前研究的重要方向。
请总结这篇文章的主要观点，并提供你对AI未来发展的看法。"""

# 代码解释任务 (~600 tokens input, 需要长输出)
long_prompt_2 = """请详细解释以下Python代码的每一行，并说明其设计模式和优化建议：

```python
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time

@dataclass
class Request:
    id: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    created_at: float = field(default_factory=time.time)
    
@dataclass  
class Response:
    id: str
    text: str
    tokens_generated: int
    latency_ms: float

class BatchProcessor:
    def __init__(self, max_batch_size: int = 32, max_wait_time: float = 0.05):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests: List[Request] = []
        self.lock = asyncio.Lock()
        self.stats: Dict[str, Any] = defaultdict(int)
        
    async def add_request(self, request: Request) -> Response:
        async with self.lock:
            self.pending_requests.append(request)
            if len(self.pending_requests) >= self.max_batch_size:
                return await self._process_batch()
        await asyncio.sleep(self.max_wait_time)
        async with self.lock:
            if self.pending_requests:
                return await self._process_batch()
                
    async def _process_batch(self) -> List[Response]:
        batch = self.pending_requests[:self.max_batch_size]
        self.pending_requests = self.pending_requests[self.max_batch_size:]
        self.stats['batches_processed'] += 1
        self.stats['requests_processed'] += len(batch)
        start = time.perf_counter()
        responses = await self._execute_model(batch)
        self.stats['total_latency'] += time.perf_counter() - start
        return responses

    请从以下几个维度分析：1) 代码功能 2) 设计模式 3) 并发安全性 4) 性能优化建议 5) 潜在bug"""   

# 创意写作任务 (要求长输出)
long_prompt_3 = """写一篇2000字的科幻短篇小说，背景设定在2150年，主题是人类与AI共同探索太阳系的故事。要求：

必须有一个主角是AI机器人
故事必须发生在木星的卫星欧罗巴上
涉及到发现外星生命的情节
有悬疑和冒险元素
结尾要有哲学思考"""

english_prompt = """what is the meaning of life?"""
# main(english_prompt, use_cudagraph=False)
main(classical_prompt, use_cudagraph=True)
