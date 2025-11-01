"""
Convert Kumru GQA (Grouped Query Attention) to MHA (Multi-Head Attention)
This is required because MoA expects MHA format.
"""
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoConfig
import shutil
import os

def expand_kv_weights(weight, num_kv_heads, num_groups):
    """
    Expand GQA K/V weights to MHA by replicating each head num_groups times.

    Args:
        weight: tensor of shape [num_kv_heads * head_dim, hidden_size]
        num_kv_heads: original number of KV heads
        num_groups: how many times to replicate each KV head

    Returns:
        expanded tensor of shape [num_kv_heads * num_groups * head_dim, hidden_size]
    """
    out_features, in_features = weight.shape
    head_dim = out_features // num_kv_heads

    # Reshape to [num_kv_heads, head_dim, in_features]
    weight_reshaped = weight.reshape(num_kv_heads, head_dim, in_features)

    # Replicate each head num_groups times
    weight_expanded = weight_reshaped.repeat_interleave(num_groups, dim=0)

    # Reshape back to [num_kv_heads * num_groups * head_dim, in_features]
    weight_expanded = weight_expanded.reshape(num_kv_heads * num_groups * head_dim, in_features)

    return weight_expanded.contiguous()

def main():
    parser = argparse.ArgumentParser(description='Convert Kumru GQA to MHA')
    parser.add_argument('--model_path', type=str, default='vngrs-ai/Kumru-2B-Base',
                      help='HuggingFace model path or local directory')
    parser.add_argument('--output_path', type=str, default='./Kumru-2B-Base-MHA',
                      help='Output directory for converted model')
    args = parser.parse_args()

    print("="*60)
    print("Converting Kumru from GQA to MHA")
    print("="*60)

    # Load config
    print(f"\n1. Loading config from: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path)
    print(f"   Original: {config.num_attention_heads} attention heads, {config.num_key_value_heads} KV heads")

    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    num_groups = num_heads // num_kv_heads
    print(f"   Expansion factor: {num_groups}x (4 KV heads -> 16 KV heads)")

    # Load GQA model
    print(f"\n2. Loading GQA model...")
    gqa_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    print(f"   Model loaded successfully")

    # Expand K/V projections in-place
    print(f"\n3. Expanding K/V projections in-place for {len(gqa_model.model.layers)} layers...")

    for i, layer in enumerate(gqa_model.model.layers):
        # Expand K projection
        k_weight_expanded = expand_kv_weights(layer.self_attn.k_proj.weight.data, num_kv_heads, num_groups)
        layer.self_attn.k_proj = torch.nn.Linear(
            in_features=layer.self_attn.k_proj.in_features,
            out_features=k_weight_expanded.shape[0],
            bias=False
        )
        layer.self_attn.k_proj.weight.data = k_weight_expanded

        # Expand V projection
        v_weight_expanded = expand_kv_weights(layer.self_attn.v_proj.weight.data, num_kv_heads, num_groups)
        layer.self_attn.v_proj = torch.nn.Linear(
            in_features=layer.self_attn.v_proj.in_features,
            out_features=v_weight_expanded.shape[0],
            bias=False
        )
        layer.self_attn.v_proj.weight.data = v_weight_expanded

        if (i + 1) % 5 == 0 or (i + 1) == len(gqa_model.model.layers):
            print(f"   Processed {i + 1}/{len(gqa_model.model.layers)} layers")

    # Update config
    print(f"\n4. Updating model config...")
    gqa_model.config.num_key_value_heads = num_heads
    mha_model = gqa_model

    # Save converted model
    print(f"\n5. Saving MHA model to: {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    mha_model.save_pretrained(args.output_path)
    print(f"   Model saved")

    # Copy tokenizer files manually (excluding broken tokenizer.json)
    print(f"\n6. Copying tokenizer files...")
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--vngrs-ai--Kumru-2B-Base")

    if os.path.exists(cache_dir):
        # Find the snapshot directory
        snapshots_dir = os.path.join(cache_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshot = os.listdir(snapshots_dir)[0]
            snapshot_path = os.path.join(snapshots_dir, snapshot)

            # Copy tokenizer files (skip tokenizer.json which is corrupted)
            for file in ['tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja']:
                src = os.path.join(snapshot_path, file)
                if os.path.exists(src):
                    shutil.copy2(src, args.output_path)
                    print(f"   Copied {file}")

    print("\n" + "="*60)
    print("Conversion complete!")
    print(f"MHA model saved to: {args.output_path}")
    print(f"Config: {num_heads} attention heads, {num_heads} KV heads")
    print("="*60)

if __name__ == "__main__":
    main()
