"""Llama 3.2 backend with cross-attention injection and LoRA adapters.

Requires optional dependencies: transformers, peft, sentencepiece.
"""

import torch
import torch.nn as nn

from src.llm.backend import BaseLLMBackend

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


class TopoCrossAttention(nn.Module):
    """Cross-attention module injected into a Llama decoder layer.

    Inserted after self-attention at a specific layer to allow the LLM
    to attend over topological memory from the TopoBridge encoder.
    """

    def __init__(self, hidden_dim: int = 2048, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, hidden_states: torch.Tensor, topo_memory: torch.Tensor
    ) -> torch.Tensor:
        """Apply cross-attention to hidden states.

        Args:
            hidden_states: (1, seq_len, hidden_dim) from self-attn output.
            topo_memory: (1, N, hidden_dim) from TopoBridge encoder.

        Returns:
            (1, seq_len, hidden_dim) with topological information mixed in.
        """
        attn_out, _ = self.cross_attn(hidden_states, topo_memory, topo_memory)
        return self.norm(hidden_states + attn_out)


class LlamaBackend(nn.Module, BaseLLMBackend):
    """Llama 3.2 1B/3B backend with frozen weights, cross-attention, and LoRA.

    Architecture:
    - Frozen Llama base model (bf16 for memory efficiency)
    - TopoCrossAttention inserted after self-attention at cross_attn_layer
    - LoRA adapters on Q/V projections at the cross-attention layer
    - Gradient checkpointing to reduce activation memory

    Requires transformers and peft packages.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        cross_attn_layer: int = 8,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        hidden_dim: int = 2048,
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for LlamaBackend. "
                "Install with: pip install transformers sentencepiece"
            )

        self.model_name = model_name
        self.cross_attn_layer = cross_attn_layer
        self.hidden_dim = hidden_dim

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load frozen model in bf16 to halve memory footprint
        self.llama = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=None,
        )
        for param in self.llama.parameters():
            param.requires_grad = False

        # Insert cross-attention at the specified layer
        self.topo_cross_attn = TopoCrossAttention(hidden_dim)

        # Apply LoRA adapters if peft is available
        if HAS_PEFT:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                layers_to_transform=[cross_attn_layer],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llama = get_peft_model(self.llama, lora_config)

        # Enable gradient checkpointing to reduce activation memory.
        # _topo_memory is NOT cleared after forward() so recomputation
        # during backward produces identical tensors.
        self.llama.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Store for use in forward hook
        self._topo_memory = None
        self._register_cross_attn_hook()

    def _get_decoder_layers(self):
        """Navigate model hierarchy to find decoder layers (handles PEFT wrapping)."""
        # Try common paths: unwrapped, PEFT-wrapped, nested
        for path in [
            lambda: self.llama.model.model.layers,      # PeftModel → CausalLM → LlamaModel
            lambda: self.llama.model.layers,             # CausalLM → LlamaModel (no PEFT)
            lambda: self.llama.base_model.model.model.layers,  # older PEFT structure
        ]:
            try:
                layers = path()
                if hasattr(layers, '__getitem__'):
                    return layers
            except AttributeError:
                continue
        raise AttributeError("Cannot find decoder layers in model hierarchy")

    def _register_cross_attn_hook(self):
        """Register a forward hook on the target layer to inject cross-attention."""
        layers = self._get_decoder_layers()
        target_layer = layers[self.cross_attn_layer]

        def hook(module, input, output):
            if self._topo_memory is None:
                return output
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Cross-attn is fp32 for stability; cast bf16→fp32→bf16
            orig_dtype = hidden.dtype
            hidden = self.topo_cross_attn(
                hidden.float(), self._topo_memory.float()
            ).to(orig_dtype)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        target_layer.register_forward_hook(hook)

    def forward(
        self,
        prefix_tokens: torch.Tensor,
        topo_memory: torch.Tensor,
        task_text: str | None = None,
    ) -> torch.Tensor:
        """Run Llama forward with prefix tokens and cross-attention.

        Args:
            prefix_tokens: (num_prefix, hidden_dim) from TopoBridge encoder.
            topo_memory: (N, hidden_dim) projected node embeddings.
            task_text: Optional text prompt for the task.

        Returns:
            hidden_states: (seq_len, hidden_dim) from the final layer.
        """
        device = prefix_tokens.device

        # Tokenize text prompt (or use minimal tokens)
        if task_text:
            text_inputs = self.tokenizer(
                task_text, return_tensors="pt", padding=True, truncation=True,
                max_length=64,
            ).to(device)
            text_embeds = self.llama.get_input_embeddings()(text_inputs["input_ids"])
            # text_embeds: (1, text_len, hidden_dim)
        else:
            # Minimal single token
            bos = torch.tensor([[self.tokenizer.bos_token_id]], device=device)
            text_embeds = self.llama.get_input_embeddings()(bos)

        # Prepend prefix tokens to text embeddings
        prefix_3d = prefix_tokens.unsqueeze(0)  # (1, K, hidden_dim)
        input_embeds = torch.cat([prefix_3d, text_embeds], dim=1)  # (1, K+text_len, hidden_dim)

        # Cast to model dtype (bf16) if needed — TopoBridge outputs fp32
        model_dtype = next(self.llama.parameters()).dtype
        input_embeds = input_embeds.to(dtype=model_dtype)

        # Set topo_memory for cross-attention hook (kept alive for grad checkpoint recomputation)
        self._topo_memory = topo_memory.unsqueeze(0).to(dtype=model_dtype)  # (1, N, hidden_dim)

        # Forward pass through Llama
        outputs = self.llama(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
        )

        # NOTE: _topo_memory intentionally NOT cleared here.
        # Gradient checkpointing replays the forward pass during backward;
        # the hook needs _topo_memory to still be set for identical recomputation.
        # It gets overwritten on the next forward() call anyway.

        # Return last hidden state, squeeze batch dim, cast back to fp32
        hidden_states = outputs.hidden_states[-1].squeeze(0).float()  # (seq_len, hidden_dim)
        return hidden_states
