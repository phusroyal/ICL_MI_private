from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_name):
    if model_name in ['gpt2-xl']:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # setup tokenizer
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise NotImplementedError(f"model_name: {model_name}")
    return model, tokenizer