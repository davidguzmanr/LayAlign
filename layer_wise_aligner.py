
import torch
from transformers.cache_utils import DynamicCache
from torch import nn

class WeightedLinearProjector(torch.nn.Module):
    """
    Weighted Linear Projector
    (bs, num_encoder_layers, num_encoder_tokens, encoder_hidden_dim)
    -> (bs, num_encoder_tokens, num_language_layers*2, encoder_hidden_dim)
    -> (bs, num_encoder_tokens, num_language_layers*2, language_hidden_dim)
    """

    def __init__(self, config, bias=True):
        super().__init__()
        self.config = config
        # 2 for key-value dim
        self.relu = nn.ReLU()
        self.fc1 = torch.nn.Linear(config.num_encoder_layers, config.num_language_layers*2, bias=bias)
        self.fc2 = torch.nn.Linear(config.encoder_hidden_dim, config.language_hidden_dim, bias=bias)
        if config.structure == "MLP":
            self.fc3 = torch.nn.Linear(config.language_hidden_dim, config.language_hidden_dim, bias=bias)
            

    def forward(self, hidden_states):
        """
        Forward pass of the weighted linear projector.
        Input:
            hidden_states: torch.Tensor
                shape: (batch_size, num_encoder_layers, num_encoder_tokens, encoder_hidden_dim)
        Output:
            past_key_values: torch.Tensor
                shape: (batch_size, num_encoder_tokens, num_language_layers*2, language_hidden_dim)
        """
        # (bs, num_encoder_layers, num_encoder_tokens, encoder_hidden_dim), e.g. (bs, 24, 24*24, 1024)
        #-> (bs, num_encoder_tokens, encoder_hidden_dim, num_encoder_layers), e.g. (bs, 24*24, 1024, 24)
        #-> (bs, num_encoder_tokens, encoder_hidden_dim, num_language_layers*2), e.g. (bs, 24*24, 1024, 32*2)
        #-> (bs, num_encoder_tokens, num_language_layers*2, encoder_hidden_dim), e.g. (bs, 24*24, 32*2, 1024)
        past_key_values = self.fc1(hidden_states.permute(0, 2, 3, 1)).permute(0, 1, 3, 2)
        past_key_values = self.relu(past_key_values)

        # 2. encoder_hidden_dim -> language_hidden_dim
        #    (bs, num_encoder_tokens, num_language_layers*2, encoder_hidden_dim), e.g. (bs, 24*24, 32*2, 1024)
        # -> (bs, num_encoder_tokens, num_language_layers*2, language_hidden_dim), e.g. (bs, 24*24, 32*2, 4096)
        past_key_values = self.fc2(past_key_values)
        if self.config.structure == "MLP":
            past_key_values = self.relu(past_key_values)

            past_key_values = self.fc3(past_key_values)
        return past_key_values




class EncoderAligner(torch.nn.Module):

    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config

        # initialize encoder hidden states projector
        self.projector = WeightedLinearProjector(config)
        # self.gate = nn.Parameter(torch.zeros(config.num_language_layers))

    def reshape_encoder_hidden_states(self, hidden_states):
        """
        Update hidden state of the encoder encoder.
        Input:
            hidden_states: torch.Tensor or List[torch.Tensor]
                shape: (batch_size, num_encoder_layers, num_encoder_tokens, encoder_hidden_dim)
                or List of encoder layer hidden states in (batch_size, num_encoder_tokens, encoder_hidden_dim)
        Return:
            hidden_states: torch.Tensor
                shape: (batch_size, num_encoder_layers, num_encoder_tokens, encoder_hidden_dim)
        """
        if type(hidden_states) in (tuple, list):
            # list or tuple of encoder layer hidden states
            # stack to tensor, (num_encoder_layers, batch_size, num_encoder_tokens, encoder_hidden_dim)
            hidden_states = torch.stack(hidden_states)
            # reshape to tensor, (batch_size, num_encoder_layers , num_encoder_tokens, encoder_hidden_dim)
            hidden_states = hidden_states.permute(1, 0, 2, 3)

        # assert dim match except for batch_size
        assert hidden_states.shape[1] == self.config.num_encoder_layers, f"hidden_states shape {hidden_states.shape} does not match config {self.config}"
        assert hidden_states.shape[3] == self.config.encoder_hidden_dim, f"hidden_states shape {hidden_states.shape} does not match config {self.config}"
        return hidden_states

    def reshape_past_key_values(self, past_key_values):
        """
        Reshape past_key_values to list of past_key_values for each language layer.
        Input:
            past_key_values: torch.Tensor
                shape: (batch_size, num_encoder_tokens, language_hidden_dim)
        Return:
            past_key_values: tuple(tuple(torch.FloatTensor))
                shape: num_language_layers * 2 * (batch_size, num_attention_heads,
                num_encoder_tokens, language_hidden_dim // num_attention_heads)
        """
        # divide hidden_dim with num_attention_heads
        # -> (bs, num_encoder_tokens, num_language_layers*2, language_hidden_dim )
        # e.g. (bs, 24*24, 64, 4096)
        batchsize = past_key_values.shape[0]
        past_key_values = past_key_values.view(
            batchsize,
            -1,
            self.config.num_language_layers * 2,
            self.config.language_hidden_dim,
        )
        # layer_coefficients = self.gate.repeat_interleave(2)
        # layer_coefficients = layer_coefficients.view(1, 1, self.config.num_language_layers * 2, 1, 1)
        # past_key_values = past_key_values * layer_coefficients
        if self.config.num_transformer_submodules == 2:
            past_key_values = torch.cat([past_key_values, past_key_values], dim=2)
        #(bs, num_encoder_tokens, num_language_layers*2, language_hidden_dim )
        # -> num_language_layers * (2, bs, num_attention_heads,
        #    num_encoder_tokens, language_hidden_dim//num_attention_heads)
        # e.g. 32 * (2, bs, 576, 4096), 2 stands for key and value
        past_key_values = past_key_values.permute([2, 0, 1, 3]).split(
            self.config.num_transformer_submodules * 2
        )

        # -> num_language_layers * 2 * (bs, num_attention_heads,
        #    num_encoder_tokens, language_hidden_dim//num_attention_heads)
        # 32 * 2 * (bs, 576, 4096)
        # ref: https://github.com/huggingface/transformers/blob/e73a97a2b338fd4bf3d97034b37dfcb29de0cb25/src/transformers/models/llama/modeling_llama.py#L818
        past_key_values = tuple(
            tuple(t.squeeze(0) for t in torch.split(past_key_value, 1, dim=0))
            for past_key_value in past_key_values
        )
        return past_key_values

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, hidden_states, *args, **kwargs):
        """
        Forward pass of the encoder prefix encoder.
        Input:
            hidden_states: torch.Tensor or List[torch.Tensor]
                shape: (batch_size, num_encoder_layers, num_encoder_tokens, encoder_hidden_dim)
                or List of encoder layer hidden states in (batch_size, num_encoder_tokens, encoder_hidden_dim)
        Return:
            past_key_values: DynamicCache(tuple(tuple(torch.FloatTensor)))
                shape: num_language_layers * 2 * (batch_size, num_attention_heads,
                num_encoder_tokens, language_hidden_dim // num_attention_heads)
        """
        hidden_states = self.reshape_encoder_hidden_states(hidden_states)

        # past_key_values: (batch_size, num_encoder_tokens, num_language_layers*2, language_hidden_dim)
        # e.g. (4, 24*24, 32*2, 4096), 2 stands for key and value
        past_key_values = self.projector.forward(hidden_states, *args, **kwargs)

        # num_language_layers * (2, bs, num_attention_heads,
        # num_encoder_tokens, language_hidden_dim//num_attention_heads)
        # e.g. 32 * (2, bs, 32, 576, 128), 2 stands for key and value
        past_key_values = self.reshape_past_key_values(past_key_values)
        return past_key_values
        #return DynamicCache.from_legacy_cache(past_key_values)
