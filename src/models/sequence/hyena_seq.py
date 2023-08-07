import math
from functools import partial

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth

from einops import rearrange

from src.utils import instantiate
import src.utils.registry as registry

from src.models.sequence.fno.tfno import FNO1d

class LinearResidual(nn.Linear):
    """Wrap nn.Linear to return the residual as well. For compatibility with FusedDense.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input), input

class SelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum('bthd,bshd->bhts', q, k * softmax_scale)
        if key_padding_mask is not None:
            padding_mask = torch.full((batch_size, seqlen), -10000.0, dtype=scores.dtype,
                                      device=scores.device)
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + rearrange(padding_mask, 'b s -> b 1 1 s')
        if causal:
            # "triu_tril_cuda_template" not implemented for 'BFloat16'
            # So we have to construct the mask in float
            causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.einsum('bhts,bshd->bthd', attention_drop, v)
        return output

class MHA(nn.Module):
    """Multi-head self-attention and cross-attention
    """

    def __init__(self, embed_dim, num_heads, bias=True, dropout=0.0,
                 softmax_scale=None, causal=False, layer_idx=None, dwconv=False,return_residual=False,device=None, dtype=None) -> None:
        """
            return_residual: whether to return the input x along with the output. This is for
                performance reason: for post-norm architecture, returning the input allows us
                to fuse the backward of nn.Linear with the residual connection.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.return_residual = return_residual

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads

        linear_cls = nn.Linear
        linear_resid_cls = LinearResidual
        inner_attn_cls =  SelfAttention

        if not self.return_residual:
            self.Wqkv = linear_cls(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        else:
            self.Wqkv = linear_resid_cls(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        if self.dwconv:
            self.dwconv_qkv = nn.Conv1d(3 * embed_dim, 3 * embed_dim, kernel_size=3, padding=2,
                                        groups=3 * embed_dim)

        self.inner_attn = inner_attn_cls(causal=causal, softmax_scale=softmax_scale,
                                         attention_dropout=dropout)

        # output projection always have the bias (for now)
        self.out_proj = linear_cls(embed_dim, embed_dim, **factory_kwargs)

    def forward(self, x, key_padding_mask=None, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """

        kwargs = ({'key_padding_mask': key_padding_mask, **kwargs})

        if not self.return_residual:
            qkv = self.Wqkv(x)
        else:
            qkv, x = self.Wqkv(x)
        if self.dwconv:
            qkv = rearrange(self.dwconv_qkv(rearrange(qkv, 'b s d -> b d s'))[..., :-2],
                            'b d s -> b s d').contiguous()
        qkv = rearrange(qkv, '... (three h d) -> ... three h d', three=3, d=self.head_dim)

        context = self.inner_attn(qkv, **kwargs)

        out = self.out_proj(rearrange(context, '... h d -> ... (h d)'))
        return out if not self.return_residual else (out, x)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, activation=F.gelu,
                 return_residual=False, device=None, dtype=None):
        """
        From https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/modules/mlp.py
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)

class Block(nn.Module):

    def __init__(self, dim, mixer_cls=None, mlp_cls=None, norm_cls=nn.LayerNorm,
                 dropout_cls=nn.Dropout, prenorm=True, resid_dropout1=0., resid_dropout2=0.,
                 drop_path1=0., drop_path2=0., 
                 return_residual=False,
                 residual_in_fp32=False):
        """
        From https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/modules/block.py
        For prenorm=True, this Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Dropout -> Add -> LN -> MHA -> Dropout -> Add -> LN -> MLP, returning both
        the hidden_states (output of the MLP) and the residual.
        This is for performance reasons, as we can fuse the dropout, add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        For prenorm=False, this Block has the same structure as a regular postnorm Transformer
        block: MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add -> LN.
        return_residual: whether each of the sub-layers (mixer and mlp) will return the residual.
        This is for performance reason: for post-norm architecture, returning the input allows us
        to fuse the backward of nn.Linear with the residual connection.
        """
        super().__init__()
        self.prenorm = prenorm
        self.return_residual = return_residual
        self.residual_in_fp32 = residual_in_fp32
        if self.residual_in_fp32:
            assert self.prenorm, 'residual_in_fp32 is only compatible with prenorm=True'
        if mixer_cls is None:
            mixer_cls = partial(MHA, num_heads=dim // 64)
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * dim)
        self.mixer = mixer_cls(dim)
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1, mode='row')
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        #TODO hardcoded FNO
        """
        self.mlp = FNO1d(n_modes_height=16, hidden_channels=64,
                in_channels=1, out_channels=1)
        """
        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(resid_dropout2)
            self.drop_path2 = StochasticDepth(drop_path2, mode='row')
            self.norm2 = norm_cls(dim)

    def forward(self, hidden_states, residual = None,
                mixer_subset=None, mixer_kwargs=None):
        r"""Pass the input through the encoder layer.
        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """
        if self.prenorm:
            dropped = self.drop_path1(self.dropout1(hidden_states))
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
            if mixer_kwargs is None:
                mixer_kwargs = {}
            if mixer_subset is not None:
                mixer_kwargs['mixer_subset'] = mixer_subset
            hidden_states = self.mixer(hidden_states, **mixer_kwargs)
            if mixer_subset is not None:
                residual = residual[:, mixer_subset]
            if not isinstance(self.mlp, nn.Identity):
                dropped = self.drop_path2(self.dropout2(hidden_states))
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
                hidden_states = self.mlp(hidden_states)
                # FNO implementation
                # FNO input should be shape = (batch, 1, 1024)
                # batched_shape = hidden_states.shape
                # hidden_states = self.mlp(hidden_states.reshape(-1,1,batched_shape[-1])).reshape(batched_shape)
            return hidden_states, residual
        else:
            assert residual is None
            mixer_out = self.mixer(
                hidden_states, **(mixer_kwargs if mixer_kwargs is not None else {})
            )
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_states = mixer_out

            hidden_states = self.norm1((self.drop_path1(self.dropout1(mixer_out))
                                        + hidden_states).to(dtype=self.norm1.weight.dtype))

            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out

                hidden_states = self.norm2((self.drop_path2(self.dropout2(mlp_out))
                                            + hidden_states).to(dtype=self.norm2.weight.dtype))

            return hidden_states

def create_mixer_cls(layer=None,
                     attn_layer_idx=None, attn_cfg=None, layer_idx=None,
                     device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    if attn_layer_idx is not None and layer_idx in attn_layer_idx:
        causal = True if attn_cfg is None else attn_cfg.pop('causal', True)

        mha_cls = MHA

        mixer_cls = partial(mha_cls, causal=causal, layer_idx=layer_idx,
                            **(attn_cfg if attn_cfg is not None else {}),**factory_kwargs)
    else:
        mixer_cls = instantiate(registry.layer, layer, partial=True, layer_idx=layer_idx, **factory_kwargs)
    return mixer_cls


def create_mlp_cls(d_model, d_inner=None, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    inner_dim = d_inner if d_inner is not None else 4 * d_model

    mlp_cls = partial(Mlp, hidden_features=inner_dim,
                          activation=partial(F.gelu, approximate='tanh'), **factory_kwargs)

    return mlp_cls


def create_block(d_model, d_inner=None,
                 layer=None, attn_layer_idx=None,
                 attn_cfg=None, layer_norm_epsilon=1e-5,
                 resid_dropout1=0.0, resid_dropout2=0.0, residual_in_fp32=False,
                 layer_idx=None,
                 device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    mixer_cls = create_mixer_cls(layer=layer,
                                 attn_layer_idx=attn_layer_idx,
                                 attn_cfg=attn_cfg, layer_idx=layer_idx,
                                 **factory_kwargs)
    mlp_cls = create_mlp_cls(d_model, d_inner=d_inner,
                             **factory_kwargs)
    norm_cls = partial(nn.LayerNorm, eps=layer_norm_epsilon, **factory_kwargs)
    block = Block(d_model, mixer_cls, mlp_cls, norm_cls=norm_cls,
                  prenorm=True, resid_dropout1=resid_dropout1, resid_dropout2=resid_dropout2,residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True,
                  glu_act=False):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
            # If using GLU activation for now, we scale the std by 2
            elif name in ["output_linear.0.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                if not glu_act:
                    nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))
                else:
                    out_features = p.shape[0]
                    # Multiplying the first half of the matrix by 2 since sigmoid scales it down by 0.5
                    # on average.
                    nn.init.normal_(p[:out_features // 2], mean=0.0, std=initializer_range / math.sqrt(2 * n_layer) * 2)

class LMBackbone(nn.Module):

    def __init__(self, d_model: int, n_layer: int, d_inner: int,
                 process_group=None, layer=None,
                 attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0,
                 resid_dropout: float = 0.0, embed_dropout: float = 0.1,
                 layer_norm_epsilon: float = 1e-5, initializer_cfg=None,residual_in_fp32=False,
                 device=None, dtype=None, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.process_group = process_group
        self.residual_in_fp32 = residual_in_fp32

        self.layers = nn.ModuleList([create_block(
            d_model, d_inner=d_inner,
            layer=layer, attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg, layer_norm_epsilon=layer_norm_epsilon,
            resid_dropout1=embed_dropout if i == 0 else resid_dropout,
            resid_dropout2=resid_dropout, residual_in_fp32=residual_in_fp32,layer_idx=i,
            **factory_kwargs,
        ) for i in range(n_layer)])

        self.drop_f = nn.Dropout(resid_dropout)
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_epsilon, **factory_kwargs)

        self.apply(partial(_init_weights, n_layer=n_layer,
                           **(initializer_cfg if initializer_cfg is not None else {})))

    def forward(self, input_ids, position_ids=None):
        hidden_states = input_ids
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        dropped = self.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))

        return hidden_states


class HyenaSequenceModel(nn.Module):

    def __init__(self, d_model: int, n_layer: int, d_inner: int,
                 layer=None,
                 attn_layer_idx=None, attn_cfg=None, max_position_embeddings=0,
                 resid_dropout: float = 0.0, embed_dropout: float = 0.1, pos_dropout: float = 0.1,
                 layer_norm_epsilon: float = 1e-5, initializer_cfg=None,residual_in_fp32=False,
                 pad_vocab_size_multiple: int = 1, n_fourier_modes: int = 64,
                 device=None, dtype=None, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.backbone = LMBackbone(
            d_model=d_model, n_layer=n_layer, d_inner=d_inner,
            layer=layer, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
            max_position_embeddings=max_position_embeddings,
            resid_dropout=resid_dropout, embed_dropout=embed_dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_cfg=initializer_cfg, residual_in_fp32=residual_in_fp32,
            **factory_kwargs, **kwargs
        )
        self.positional = PositionalEncoding(
            d_model=d_model,
            dropout=pos_dropout,
            device=device,
        )
        self.proj = torch.nn.Linear(
            in_features=d_model,
            out_features=32,
        )
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.proj2 = torch.nn.Linear(
            in_features=32,
            out_features=1,
        )
        self.fno = FNO1d(
            n_modes_height=n_fourier_modes,
            hidden_channels=64,
            in_channels=1,
            out_channels=1 # takes in (batch x 1 x seqlen)
        )
        # self.lm_head = nn.Linear(d_model, 1, bias=False, **factory_kwargs)

        # self.final_pool = nn.Conv1d(
        #         in_channels=d_model,
        #         out_channels=d_model,
        #         kernel_size=l_max, # not defined? TODO
        #     )

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, n_layer=n_layer,
                           **(initializer_cfg if initializer_cfg is not None else {})))

    def forward(self, input_ids, pde_params, position_ids=None, state=None): # state for the repo interface
        # print(input_ids.shape)
        hidden_states = self.backbone(input_ids, position_ids=position_ids)
        encoding_vector = torch.tanh(torch.mean(hidden_states, axis=1)) # batch_size, d_model
        pde_params_pred = self.relu1(self.proj2(
            self.relu2(self.proj(encoding_vector))
        ))
        return pde_params_pred, None

        # print(hidden_states.shape)
        # This encoding method is non-causal and thus is cheating on the intermediate predictions
        # TODO double-check this code is correct
        """
        encoding_vector = torch.tanh(torch.mean(hidden_states[:, :, :], axis=1)) # batch_size x d_model
        # print(encoding_vector.shape)
        fno_in = input_ids[:, -1:, :] # batch_size x num_examples x d_model
        encoding_vector = encoding_vector.unsqueeze(1).repeat((1,fno_in.shape[1],1)) # batch_size x num_examples x d_model
        fno_in = fno_in + encoding_vector
        fno_in = rearrange(fno_in, 'b n d -> (b n) 1 d')
        fno_out = self.fno(fno_in)
        fno_out = rearrange(fno_out, '(b n) 1 d -> b n d', b=input_ids.shape[0])
        """

        # fno_in = (encoding_vector + input_ids[:, -1, :]).unsqueeze(1) # batch_size x 1 x d_model

        # TODO: curriculum-like setup. Compute cumulative estimates of 
        # encoding_vector = torch.cumsum(hidden_states, dim=1)
        # # Compute cumulative lengths
        # encoding_vector_lens = torch.arange(1, hidden_states.shape[1]+1, device=encoding_vector.device) # num_examples
        # encoding_vector = encoding_vector / encoding_vector_lens[None, :, None]
        # print(f"encoding vector: {encoding_vector.shape}")
        # print(f"input: {input_ids.shape}")
        # fno_in = input_ids[:, 0::2, :] + encoding_vector[:, 0::2, :]

        # Extra supervision on FNO
        # FNO baseline
        """
        fno_in = input_ids[:, 0::2, :] # batch_size, num_examples, d_model
        pde_params = self.positional(pde_params) # batch_size, d_model
        # num_examples - 1 so that the final embedding is the ICL embedding from Hyena
        pde_params = pde_params.unsqueeze(1).repeat(1, fno_in.shape[1]-1, 1) # batch_size, num_examples-1, d_model
        encoding_vector = torch.tanh(torch.mean(hidden_states[:, :, :], axis=1)) # batch_size, d_model
        print(f"encoding vector: {encoding_vector.shape}")
        pde_params = torch.cat([pde_params, encoding_vector.unsqueeze(1)], dim=1) # batch_size, num_examples, d_model
        print(f"fno_in: {fno_in.shape}")
        print(f"pde_params: {pde_params.shape}")
        fno_in = fno_in + pde_params
        fno_in = rearrange(fno_in, 'b n d -> (b n) 1 d')
        fno_out = self.fno(fno_in)
        fno_out = rearrange(fno_out, '(b n) 1 d -> b n d', b=input_ids.shape[0])
        """

        # FNO baseline
        # fno_in = input_ids[:, -1, :].unsqueeze(1)
        # fno_out = self.fno(fno_in)

        # lm_logits = self.lm_head(hidden_states)
        # CausalLMOutput = namedtuple('CausalLMOutput', ['logits'])
        # return CausalLMOutput(logits=lm_logits), None

        # return hidden_states, None

        return fno_out, None

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, device="cuda"):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.device = device

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size]``
        Output:
            pe: Tensor, shape ``[batch_size, d_model]
        """
        r = torch.arange(1, self.d_model//2+1).cuda()
        r = (-math.log(10000.0) / r).cuda()
        x_repeat = x.unsqueeze(-1).repeat(*([1]*len(x.shape)), self.d_model//2).float().cuda()
        div_term = x.unsqueeze(-1).repeat(*([1]*len(x.shape)), self.d_model//2).float().cuda()
        div_term[..., :] *= r
        div_term = torch.exp(div_term)
        pe = torch.zeros(*x.shape, self.d_model).cuda()
        pe[..., 0::2] = torch.sin(x_repeat * div_term)
        pe[..., 1::2] = torch.cos(x_repeat * div_term)

        return self.dropout(pe)

class FNOBaseline(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1,
                 fno_nmodes: int = 64, fno_nhidden: int = 64,
                 device=None, dtype=None, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.positional = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            device=device,
        )
        self.fno = FNO1d(
            n_modes_height=fno_nmodes,
            hidden_channels=fno_nhidden,
            in_channels=1,
            out_channels=1 # takes in (batch x 1 x seqlen)
        )

        # Initialize weights and apply final processing
        # self.apply(partial(_init_weights, n_layer=n_layer,
        #                    **(initializer_cfg if initializer_cfg is not None else {})))

    def forward(self, input_ids, pde_params, position_ids=None, state=None): # state for the repo interface

        # fno_in = (encoding_vector + input_ids[:, -1, :]).unsqueeze(1) # batch_size x 1 x d_model
        # FNO baseline
        fno_in = input_ids[:, 0::2, :] # batch_size, num_examples, d_model
        # print(fno_in.shape)
        pde_params = self.positional(pde_params)
        # print(pde_params.shape)
        pde_params = pde_params.unsqueeze(1).repeat(1, fno_in.shape[1], 1)
        fno_in = rearrange(fno_in, 'b n d -> (b n) 1 d')
        pde_params = rearrange(pde_params, 'b n d -> (b n) 1 d')
        fno_out = self.fno(fno_in+pde_params)
        fno_out = rearrange(fno_out, '(b n) 1 d -> b n d', b=input_ids.shape[0])

        # return hidden_states, None
        return fno_out, None

# class HyenaSequenceModel(nn.Module):

#     def __init__(
#         self,
#         d_model,
#         l_max,
#         n_layer=2,
#         non_linearity=F.gelu,
#         skip="identity",
#         order=2, 
#         filter_order=64,
#         num_heads=1, 
#         inner_factor=1,
#         num_blocks=1, 
#         fused_bias_fc=False,
#         outer_mixing=False,
#         dropout=0.0,  
#         filter_dropout=0.0, 
#         filter_cls='hyena-filter',
#         post_order_ffn=False,
#         jit_filter=False, 
#         short_filter_order=3, 
#         activation="id",
#         return_state=False,
#         **filter_args,
#     ):

#         self.n_layer = n_layer
#         self.return_state = return_state
#         self.non_linearity = non_linearity

#         if skip == "identity":
#             skip_connection = nn.Identity()

#         self.hyena_layers = nn.ModuleList([
#             HyenaOperator(
#                 d_model,
#                 l_max,
#                 order=2, 
#                 filter_order=64,
#                 num_heads=1, 
#                 inner_factor=1,
#                 num_blocks=1, 
#                 fused_bias_fc=False,
#                 outer_mixing=False,
#                 dropout=0.0,  
#                 filter_dropout=0.0, 
#                 filter_cls='hyena-filter',
#                 post_order_ffn=False,
#                 jit_filter=False, 
#                 short_filter_order=3, 
#                 activation="id",
#                 return_state=False,
#                 **filter_args,
#             )
#             for _ in range(self.n_layer)
#         ])

#         self.skip_layers = nn.ModuleList([
#             skip_connection for _ in range(self.n_layer)
#         ])

#     def forward(self, u, *args, **kwargs):
#         for i in range(self.n_layer):
#             if self.return_state:
#                 u_hyena, _ = self.hyena_layers[i](u)
#             else:
#                 u_hyena = self.hyena_layers[i](u)
#             u_skip = self.skip_layers[i](u)
#             u = u_hyena + u_skip
#             if i < self.n_layer-1:
#                 u = self.non_linearity(u)
#         if self.return_state:
#             return u, None
#         return u
