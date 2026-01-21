#!/usr/bin/env python3
"""
PaddleFormers VLM standalone test.
Tests AutoModel loading for VLM models to verify architectures attribute fix.
"""
import time
import paddle
from paddleformers.transformers import AutoModel, AutoConfig, AutoProcessor

def test_config_architectures():
    """Test that VLM config properly exposes architectures attribute."""
    model_dir = "/home/disk2/yangjiajie/Qwen/Qwen3-VL-2B-Instruct"
    
    print("=" * 60)
    print("TEST 1: Config architectures attribute")
    print("=" * 60)
    
    config = AutoConfig.from_pretrained(model_dir)
    print(f"Config type: {type(config).__name__}")
    print(f"Model type: {config.model_type}")
    print(f"architectures: {getattr(config, 'architectures', 'NOT_FOUND')}")
    
    # Check if architectures is correctly accessible
    if config.architectures is None:
        print("\n❌ FAIL: architectures is None (attribute shadowing bug)")
        return False
    else:
        print(f"\n✅ PASS: architectures = {config.architectures}")
        return True


def test_automodel_from_config():
    """Test AutoModel.from_config() for VLM models."""
    model_dir = "/home/disk2/yangjiajie/Qwen/Qwen3-VL-2B-Instruct"
    
    print("\n" + "=" * 60)
    print("TEST 2: AutoModel.from_config() loading")
    print("=" * 60)
    
    print("\nLoading config...")
    config = AutoConfig.from_pretrained(model_dir, dtype="bfloat16")
    print(f"Config loaded: {type(config).__name__}")
    print(f"architectures: {config.architectures}")
    
    print("\nLoading model via AutoModel.from_config()...")
    try:
        load_start = time.time()
        model = AutoModel.from_config(config, dtype="bfloat16")
        load_time = time.time() - load_start
        print(f"✅ PASS: Model loaded in {load_time:.2f}s")
        print(f"Model type: {type(model).__name__}")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_full_model_loading():
    """Test full model loading with weights."""
    model_dir = "/home/disk2/yangjiajie/Qwen/Qwen3-VL-2B-Instruct"
    
    print("\n" + "=" * 60)
    print("TEST 3: Full model loading with weights")
    print("=" * 60)
    
    print(f"\nGPU Memory Before: {paddle.device.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    print("\nLoading model...")
    try:
        from paddleformers.transformers import Qwen3VLForConditionalGeneration
        load_start = time.time()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir,
            dtype="bfloat16",
            convert_from_hf=True,
        ).eval()
        load_time = time.time() - load_start
        print(f"✅ PASS: Model loaded in {load_time:.2f}s")
        print(f"Model type: {type(model).__name__}")
        print(f"GPU Memory After: {paddle.device.cuda.memory_allocated() / 1024**2:.1f} MB")
        return model
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_text_generation(model):
    """Test text-only generation."""
    model_dir = "/home/disk2/yangjiajie/Qwen/Qwen3-VL-2B-Instruct"
    
    print("\n" + "=" * 60)
    print("TEST 4: Text-only generation")
    print("=" * 60)
    
    if model is None:
        print("❌ SKIP: Model not loaded")
        return False
    
    processor = AutoProcessor.from_pretrained(model_dir)
    
    # Simple text prompt
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "你好，请简单介绍一下你自己。"}]}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pd", padding=True)
    
    print(f"Input length: {inputs['input_ids'].shape[1]} tokens")
    
    print("\nGenerating...")
    try:
        with paddle.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
            )
        
        output_text = processor.batch_decode(outputs[0], skip_special_tokens=True)[0]
        print(f"✅ PASS: Generation successful")
        print(f"\nOutput:\n{output_text}")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("PaddleFormers VLM Standalone Test")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Config architectures
    results['config_architectures'] = test_config_architectures()
    
    # Test 2: AutoModel.from_config (the actual bug we're fixing)
    results['automodel_from_config'] = test_automodel_from_config()
    
    # Test 3: Full model loading  
    model = test_full_model_loading()
    results['full_model_loading'] = model is not None
    
    # Test 4: Text generation
    if model is not None:
        results['text_generation'] = test_text_generation(model)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    return all_passed


if __name__ == "__main__":
    main()
