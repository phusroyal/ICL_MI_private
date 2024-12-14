import torch

def hook_io_b4rs(model, inputs):
    # Store the hooks and outputs
    hooks = []
    outputs = {'in': [], 'out': []}

    # Define the hook function to capture Q, K, V matrices and attention output per head
    def hook_fn_qkv(module, input, output, is_cross_attention=False):
        hidden_states = input[0]
        attention_mask = input[2] if len(input) > 2 else None
        bsz, q_len, _ = hidden_states.size()

        # Determine if cross-attention is used
        # is_cross_attention = False

        # Extract Q, K, V matrices
        if is_cross_attention:
            query = module.q_attn(hidden_states)
            key, value = module.c_attn(module.encoder_hidden_states).split(module.split_size, dim=2)
        else:
            query, key, value = module.c_attn(hidden_states).split(module.split_size, dim=2)
        
        outputs['in'].append(value)

        query = module._split_heads(query, module.num_heads, module.head_dim)
        key = module._split_heads(key, module.num_heads, module.head_dim)
        value = module._split_heads(value, module.num_heads, module.head_dim)

        # Hook into the scaled dot-product attention function
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,  # Attention mask
            dropout_p=module.attn_dropout.p if module.training else 0.0,
            is_causal=True if attention_mask is None and q_len > 1 and not is_cross_attention else False
        )

        # Reshape outputs
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, module.embed_dim)

        # Final projection
        attn_output = module.c_proj(attn_output)
        attn_output = module.resid_dropout(attn_output)

        # Capture attention output from each head
        # outputs['in'].append(hidden_states)
        # outputs['in'].append(value)
        outputs['out'].append(attn_output)

    # Register hooks on each layer's attention
    for layer in model.h:
        hooks.append(layer.attn.register_forward_hook(hook_fn_qkv))

    # Run the model to capture outputs
    with torch.no_grad():
        model(**inputs)

    # Remove the hooks after running the model
    for hook in hooks:
        hook.remove()

    return outputs

# # Usage
# from transformers import GPT2Model, GPT2Tokenizer
# model = GPT2Model.from_pretrained("gpt2")
# prompt = "female\tmiddle\t28 -> survival\nmale\tupper\t51 -> death\nmale\tlower\t21 ->"
# outputs = hook_qkv_and_head_outputs(model, prompt)

# # Now you can inspect the outputs dictionary
# print("Q matrices shape at different layers and heads:", [q.shape for q in outputs['Q']])
# print("K matrices shape at different layers and heads:", [k.shape for k in outputs['K']])
# print("V matrices shape at different layers and heads:", [v.shape for v in outputs['V']])
# print("Attention output shape at different layers and heads:", [ao.shape for ao in outputs['attn_output_each_head']])
