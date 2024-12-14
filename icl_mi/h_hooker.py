import torch
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer

class GPT2WithBlockIO(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.block_inputs = []
        self.block_outputs = []

        # Register hooks for each GPT2Block
        for i, block in enumerate(self.h):
            block.register_forward_hook(self._get_block_io_hook(i))

    def _get_block_io_hook(self, block_idx):
        def hook(module, input, output):
            # Save the input and output of the block
            self.block_inputs.append(input)
            self.block_outputs.append(output)
        return hook

    def forward(self, *args, **kwargs):
        # Clear previous inputs and outputs
        self.block_inputs = []
        self.block_outputs = []
        
        # Call the superclass forward method
        return super().forward(*args, **kwargs)


class GPT2WithLayerOutputs(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize dictionaries to store outputs
        self.ln1_outputs = {}
        self.ln2_outputs = {}
        self.mlp_outputs = {}
        
        # Register hooks
        for i, block in enumerate(self.h):
            block.ln_1.register_forward_hook(self._get_hook(self.ln1_outputs, i))
            block.ln_2.register_forward_hook(self._get_hook(self.ln2_outputs, i))
            block.mlp.register_forward_hook(self._get_hook(self.mlp_outputs, i))
    
    def _get_hook(self, output_dict, layer_idx):
        def hook(module, input, output):
            output_dict[layer_idx] = output
        return hook
    
    def forward(self, *args, **kwargs):
        # Clear outputs
        self.ln1_outputs.clear()
        self.ln2_outputs.clear()
        self.mlp_outputs.clear()
        
        return super().forward(*args, **kwargs)