#!/usr/bin/env python3
"""
Standalone PaddleFormers benchmark test.
Generates output to compare with FastDeploy fallback performance.
"""
import time
from paddleformers.transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import paddle
import numpy as np

def main(prompt, use_cache=True, attn_impl="eager"):
    # Model path
    # model_dir = "/home/disk2/yangjiajie/Qwen/Qwen3-0.6B"
    model_dir = "/home/disk2/yangjiajie/Qwen/Qwen3-4B"
    # model_dir = "/home/disk2/yangjiajie/Qwen/Qwen2.5-3B-Instruct"
    # model_dir = "/home/disk2/yangjiajie/microsoft/Phi-3-mini-4k-instruct"
    # model_dir = "/home/disk2/yangjiajie/PaddlePaddle/ERNIE-4.5-0.3B-PT"
    # model_dir = "/home/disk2/yangjiajie/unsloth/gemma-3-1b-it"

    print("=" * 60)
    print("PaddleFormers Standalone Benchmark")
    print("=" * 60)

    # Check GPU memory before loading
    print("\n=== GPU Memory Before Model Load ===")
    print(f"Allocated: {paddle.device.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"Reserved:  {paddle.device.cuda.memory_reserved() / 1024**2:.1f} MB")

    # Load model and tokenizer
    print("\nLoading tokenizer and config...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print("Loading model (bfloat16)...")
    print(f">>> attn_implementation: {attn_impl}")
    load_start = time.time()
    config = AutoConfig.from_pretrained(model_dir, dtype="bfloat16", 
        # fuse_attention_qkv=True,
        use_cache=True,
        # fuse_attention_ffn=True,
        _attn_implementation="sdpa",)
    # config.fuse_rms_norm = True    

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        dtype="bfloat16", 
        config=config,  
        convert_from_hf=True,
        # load_checkpoint_format="flex_checkpoint",
    ).eval()
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    print(f"  [DEBUG] AFTER load: q_norm mean={float(model.model.layers[0].self_attn.q_norm.weight.mean()):.6f}, std={float(model.model.layers[0].self_attn.q_norm.weight.std()):.6f}")
    print(f"  [DEBUG] AFTER load: k_norm mean={float(model.model.layers[0].self_attn.k_norm.weight.mean()):.6f}, std={float(model.model.layers[0].self_attn.k_norm.weight.std()):.6f}")
    print(f"  [DEBUG] AFTER load: q_proj mean={float(model.model.layers[0].self_attn.q_proj.weight.mean()):.6f}, std={float(model.model.layers[0].self_attn.q_proj.weight.std()):.6f}")
    print(f"  [DEBUG] AFTER load: k_proj mean={float(model.model.layers[0].self_attn.k_proj.weight.mean()):.6f}, std={float(model.model.layers[0].self_attn.k_proj.weight.std()):.6f}")
    print(f"  [DEBUG] AFTER load: v_proj mean={float(model.model.layers[0].self_attn.v_proj.weight.mean()):.6f}, std={float(model.model.layers[0].self_attn.v_proj.weight.std()):.6f}")
    print(f"  [DEBUG] AFTER load: o_proj mean={float(model.model.layers[0].self_attn.o_proj.weight.mean()):.6f}, std={float(model.model.layers[0].self_attn.o_proj.weight.std()):.6f}")
    print(f"  [DEBUG] AFTER load: input_layernorm mean={float(model.model.layers[0].input_layernorm.weight.mean()):.6f}, std={float(model.model.layers[0].input_layernorm.weight.std()):.6f}")
    print(f"  [DEBUG] AFTER load: post_attention_layernorm mean={float(model.model.layers[0].post_attention_layernorm.weight.mean()):.6f}, std={float(model.model.layers[0].post_attention_layernorm.weight.std()):.6f}")
    # Verify fused options were applied
    print(f">>> Config check:")
    print(f"    fused_rms_norm = {getattr(model.config, 'fused_rms_norm', 'N/A')}")
    print(f"    fuse_attention_qkv = {getattr(model.config, 'fuse_attention_qkv', 'N/A')}")
    print(f"    fuse_attention_ffn = {getattr(model.config, 'fuse_attention_ffn', 'N/A')}")

    # Check GPU memory after loading
    print("\n=== GPU Memory After Model Load ===")
    print(f"Allocated: {paddle.device.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"Reserved:  {paddle.device.cuda.memory_reserved() / 1024**2:.1f} MB")

    # Basic info
    print(f"\nTokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Model vocab size: {config.vocab_size}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")

    # Prepare prompt
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{prompt}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pd", add_special_tokens=True)
    input_ids_list = inputs["input_ids"][0].tolist()
    input_len = len(input_ids_list)
    print(f"Input token count: {input_len}")

    
    # Warmup
    print("Warmup (3 runs)...")
    for _ in range(3):
        with paddle.no_grad():
            _ = model.model(inputs["input_ids"], use_cache=True)
        paddle.device.synchronize()
    # Prefill Benchmark
    times = []
    for _ in range(10):
        paddle.device.synchronize()
        start = time.time()
        with paddle.no_grad():
            _ = model.model(inputs["input_ids"], use_cache=True)
        paddle.device.synchronize()
        times.append(time.time() - start)
    
    times = np.array(times)
    print(f"Mean: {times.mean()*1000:.2f}ms ± {times.std()*1000:.2f}ms")

    # Check GPU memory after warmup
    print("\n=== GPU Memory After Warmup ===")
    print(f"Allocated: {paddle.device.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"Reserved:  {paddle.device.cuda.memory_reserved() / 1024**2:.1f} MB")

    # Benchmark run - use max_new_tokens instead of max_new_tokens
    print("\nBenchmark run...")
    paddle.device.synchronize()
    start_time = time.time()


    print(f"\n>>> Running with use_cache={use_cache}")
    with paddle.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,  # Greedy Search
        )

    paddle.device.synchronize()
    end_time = time.time()

    # Check GPU memory after generation
    print("\n=== GPU Memory After Generation ===")
    print(f"Allocated: {paddle.device.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"Reserved:  {paddle.device.cuda.memory_reserved() / 1024**2:.1f} MB")

    # Debug: print output structure
    print(f"\n=== Output Debug ===")
    print(f"outputs type: {type(outputs)}")
    print(f"outputs[0] type: {type(outputs[0])}")
    if hasattr(outputs[0], 'shape'):
        print(f"outputs[0] shape: {outputs[0].shape}")
    else:
        print(f"outputs[0] length: {len(outputs[0])}")

    # Results - handle different output formats
    if isinstance(outputs[0], paddle.Tensor):
        output_ids = outputs[0][0].tolist() if outputs[0].dim() > 1 else outputs[0].tolist()
    else:
        output_ids = outputs[0][0].tolist() if hasattr(outputs[0][0], 'tolist') else list(outputs[0][0])

    output_len = len(output_ids)
    elapsed_time = end_time - start_time
    tokens_per_second = output_len / elapsed_time if elapsed_time > 0 else 0

    print(f"\n{'=' * 60}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    print(f"Input tokens:  {input_len}")
    print(f"Output tokens: {output_len}")
    print(f"Time elapsed:  {elapsed_time:.2f}s")
    print(f"Throughput:    {tokens_per_second:.2f} tokens/s")
    print(f"{'=' * 60}")

    # Show output
    print("\n=== Generated Output IDs ===")
    print(output_ids)
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"\n=== Generated Output ===")
    print(output_text)


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

# A/B Test: compare attention implementations
# print("\n" + "=" * 60)
# print("TEST 1: attn_implementation=sdpa (default, slow)")
# print("=" * 60)
# main(long_prompt_1, use_cache=False, attn_impl="sdpa")

print("\n\n" + "=" * 60)
print("TEST 2: attn_implementation=sdpa (should be faster)")
print("=" * 60)
english_prompt = """what is the meaning of life?"""
main(classical_prompt, use_cache=True)