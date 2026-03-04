import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple
import math

class RecursiveLanguageModelConfig(PretrainedConfig):
    model_type = "recursive_language_model"

    def __init__(
        self,
        vocab_size: int = 50257,
        embedding_dim: int = 512,
        num_layers: int = 6,
        num_attention_heads: int = 8,
        max_recursion_steps: int = 5,
        max_position_embeddings: int = 512,
        hidden_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        intermediate_size: int = 2048,
        layer_norm_eps: float = 1e-5,
        pad_token_id: int = 50256,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        simple_recursion_steps: int = 1,
        medium_recursion_steps: int = 3,
        complex_recursion_steps: int = 5,
        confidence_threshold: float = 0.8,
        use_adaptive_stopping: bool = True,
        initializer_range: float = 0.02,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.max_recursion_steps = max_recursion_steps
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.simple_recursion_steps = simple_recursion_steps
        self.medium_recursion_steps = medium_recursion_steps
        self.complex_recursion_steps = complex_recursion_steps
        self.confidence_threshold = confidence_threshold
        self.use_adaptive_stopping = use_adaptive_stopping
        self.initializer_range = initializer_range


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self.dim = dim

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    def __init__(self, config: RecursiveLanguageModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.embedding_dim // config.num_attention_heads
        self.embed_dim = config.embedding_dim

        assert self.embed_dim % self.num_heads == 0

        self.q_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.k_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.v_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.out_proj = nn.Linear(config.embedding_dim, config.embedding_dim)

        self.dropout = nn.Dropout(config.attention_dropout_prob)
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, config.max_position_embeddings)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(seq_len, hidden_states.device)
        cos = cos[None, None, :, :].expand(batch_size, self.num_heads, -1, -1)
        sin = sin[None, None, :, :].expand(batch_size, self.num_heads, -1, -1)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class FeedForward(nn.Module):
    def __init__(self, config: RecursiveLanguageModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.embedding_dim, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: RecursiveLanguageModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SequenceLevelRouter(nn.Module):
    def __init__(self, config: RecursiveLanguageModelConfig):
        super().__init__()
        self.config = config
        self.pooler = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.pooler_activation = nn.Tanh()
        self.classifier = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.embedding_dim // 2, 3)
        )

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask
        else:
            pooled = hidden_states.mean(dim=1)

        pooled = self.pooler(pooled)
        pooled = self.pooler_activation(pooled)
        complexity_logits = self.classifier(pooled)
        complexity_class = torch.argmax(complexity_logits, dim=-1)

        recursion_steps = torch.zeros_like(complexity_class)
        recursion_steps[complexity_class == 0] = self.config.simple_recursion_steps
        recursion_steps[complexity_class == 1] = self.config.medium_recursion_steps
        recursion_steps[complexity_class == 2] = self.config.complex_recursion_steps

        return complexity_logits, complexity_class, recursion_steps


class RecursionLayer(nn.Module):
    def __init__(self, config: RecursiveLanguageModelConfig):
        super().__init__()
        self.transformer_block = TransformerBlock(config)

    def forward(self, hidden_states, attention_mask=None):
        return self.transformer_block(hidden_states, attention_mask)


class RecursiveLanguageModel(PreTrainedModel):
    config_class = RecursiveLanguageModelConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: RecursiveLanguageModelConfig):
        super().__init__(config)
        self.config = config

        self.embedding_layer = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=config.pad_token_id
        )

        self.base_transformer = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.router = SequenceLevelRouter(config)
        self.recursion_layer = RecursionLayer(config)

        self.final_layer_norm = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.language_model_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        self.post_init()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (TransformerBlock, RecursionLayer)):
            module.gradient_checkpointing = value

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.embedding_layer

    def set_input_embeddings(self, value):
        self.embedding_layer = value

    def get_output_embeddings(self):
        return self.language_model_head

    def set_output_embeddings(self, new_embeddings):
        self.language_model_head = new_embeddings

    def get_attention_mask(self, input_ids):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=device)
        attention_mask[:, :, causal_mask] = float('-inf')

        padding_mask = (input_ids == self.config.pad_token_id)
        valid_mask = ~padding_mask

        if padding_mask.any():
            padding_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.masked_fill(padding_mask_expanded, float('-inf'))

        return attention_mask, valid_mask

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embedding_layer(input_ids)
        attn_mask, padding_mask = self.get_attention_mask(input_ids)

        for layer in self.base_transformer:
            hidden_states = layer(hidden_states, attn_mask)

        complexity_logits, complexity_class, recursion_steps = self.router(
            hidden_states, padding_mask
        )

        max_steps_in_batch = int(recursion_steps.max().item())

        for step in range(max_steps_in_batch):
            step_mask = (recursion_steps > step).float().unsqueeze(-1).unsqueeze(-1)
            new_hidden = self.recursion_layer(hidden_states, attn_mask)
            hidden_states = step_mask * new_hidden + (1 - step_mask) * hidden_states

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.language_model_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

            with torch.no_grad():
                loss_fct_per_sample = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                token_losses = loss_fct_per_sample(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1)
                ).view(batch_size, -1)

                valid_tokens = (shift_labels != -100).sum(dim=1).clamp(min=1)
                sample_loss = token_losses.sum(dim=1) / valid_tokens
                sample_perplexity = torch.exp(sample_loss)

                pseudo_labels = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
                pseudo_labels[sample_perplexity < 20] = 0
                pseudo_labels[(sample_perplexity >= 20) & (sample_perplexity < 50)] = 1
                pseudo_labels[sample_perplexity >= 50] = 2

            router_loss_fct = nn.CrossEntropyLoss()
            router_loss = router_loss_fct(complexity_logits, pseudo_labels)

            loss = lm_loss + 0.1 * router_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0,
                 top_p=0.9, do_sample=True, **kwargs):
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.forward(generated)
                logits = outputs.logits

            next_token_logits = logits[:, -1, :] / temperature

            if do_sample:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == self.config.eos_token_id:
                break

        return generated
