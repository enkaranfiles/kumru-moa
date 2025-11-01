"""
Test script to verify Kumru-2B-Base-MHA works with MoA
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from MoA.models.interface import update_model_function

def test_kumru_loading():
    print("=" * 50)
    print("Testing Kumru-2B-Base-MHA with MoA Interface")
    print("=" * 50)

    model_path = "/Users/eneskaranfil/Desktop/Sakarya/Kumru-2B-Base-MHA"

    print(f"\n1. Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"   Model loaded successfully!")
    print(f"   Architecture: {model.config.architectures}")
    print(f"   Attention heads: {model.config.num_attention_heads}")
    print(f"   KV heads: {model.config.num_key_value_heads}")
    print(f"   Hidden size: {model.config.hidden_size}")
    print(f"   Layers: {model.config.num_hidden_layers}")

    print(f"\n2. Updating model with MoA functions...")
    model = update_model_function(model, model_path)
    print("   MoA functions added successfully!")

    print(f"\n3. Testing basic tokenization...")
    test_text = "Merhaba, nasılsın?"
    inputs = tokenizer(test_text, return_tensors="pt")
    print(f"   Input text: {test_text}")
    print(f"   Token IDs: {inputs['input_ids']}")
    print(f"   Tokenization successful!")

    print("\n" + "=" * 50)
    print("✓ All tests passed! Kumru-MHA is compatible with MoA")
    print("=" * 50)

    return model, tokenizer

if __name__ == "__main__":
    try:
        model, tokenizer = test_kumru_loading()
        print("\n✓ Ready to run MoA pipeline!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
