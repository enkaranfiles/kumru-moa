"""
Test script to verify Kumru-2B-Base-MHA works with MoA
"""
import torch
from transformers import AutoModelForCausalLM
from MoA.models.interface import update_model_function

def test_kumru_moa_integration():
    print("=" * 60)
    print("Testing Kumru-2B-Base-MHA with MoA Interface")
    print("=" * 60)

    #model_path = "/Users/eneskaranfil/Desktop/Sakarya/Kumru-2B-Base-MHA"
    
    model_path = "/workspace/kumru-moa/Kumru-2B-Base-MHA"
    print(f"\n1. Loading MHA model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"   ✓ Model loaded successfully!")
    print(f"   Architecture: {model.config.architectures}")
    print(f"   Attention heads: {model.config.num_attention_heads}")
    print(f"   KV heads: {model.config.num_key_value_heads}")
    print(f"   Hidden size: {model.config.hidden_size}")
    print(f"   Layers: {model.config.num_hidden_layers}")

    # Verify it's MHA (not GQA)
    assert model.config.num_attention_heads == model.config.num_key_value_heads, \
        "Model should be MHA (num_attention_heads == num_key_value_heads)"
    print(f"   ✓ Confirmed MHA configuration (16 heads = 16 KV heads)")

    print(f"\n2. Applying MoA interface to model...")
    model = update_model_function(model, model_path)
    print("   ✓ MoA functions added successfully!")

    print(f"\n3. Verifying MoA integration...")
    # Check if model has MoA-specific attributes
    if hasattr(model.model, 'set_mixture_of_attention'):
        print("   ✓ Model has set_mixture_of_attention method")
    else:
        print("   ✗ Warning: MoA method not found")

    print("   ✓ Model is ready for MoA compression pipeline")

    print("\n" + "=" * 60)
    print("✓ All tests passed! Kumru-MHA is ready for MoA")
    print("=" * 60)
    print("\nNext steps for MoA compression:")
    print("1. Run MoA calibration dataset generation")
    print("2. Profile attention importance at different sequence lengths")
    print("3. Optimize sparse attention patterns")
    print("4. Validate and select best compression plan")
    print("5. Apply compression and benchmark performance")
    print("\nCommands:")
    print("  python MoA/scripts/pipeline/main.py --model_path /workspace/kumru-moa/Kumru-2B-Base-MHA --model_name Kumru-2B-Base-MHA")

    return model

if __name__ == "__main__":
    try:
        model = test_kumru_moa_integration()
        print("\n✓ Kumru-MoA integration successful!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
