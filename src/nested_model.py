import logging
from typing import List
import torch
from torch import nn
from transformers import AutoConfig

from src.base_model import BaseModel

logger = logging.getLogger(__name__)

class Biaffine(nn.Module):
    """
    Biaffine layer for scoring spans: computes scores for every start-end pair.
    """
    def __init__(self, in1_features: int, in2_features: int, out_features: int,
                 bias_x: bool = True, bias_y: bool = True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(
            out_features,
            in1_features + int(bias_x),
            in2_features + int(bias_y)
        ))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, in1), y: (batch, seq_len, in2)
        if self.bias_x:
            ones = x.new_ones(*x.shape[:-1], 1)
            x = torch.cat([x, ones], dim=-1)
        if self.bias_y:
            ones = y.new_ones(*y.shape[:-1], 1)
            y = torch.cat([y, ones], dim=-1)
        # Compute biaffine: first xW -> (batch, out, seq_len, in2)
        xW = torch.einsum('bxi,oij->boxj', x, self.weight)
        # Then contract with y -> (batch, out, seq_len, seq_len)
        scores = torch.einsum('boxj,byj->boxy', xW, y)
        # Permute to (batch, seq_len, seq_len, out)
        return scores.permute(0, 2, 3, 1)

class NestedLawTagger(BaseModel):
    """
    Nested span tagger for legal entities using a biaffine scorer.
    Outputs a score tensor of shape (batch, seq_len, seq_len, num_types).
    """
    def __init__(
        self,
        bert_model_name: str = "aubmindlab/bert-base-arabertv2",
        label_list: List[str] = None,
        type_list: List[str] = None,
        dropout: float = 0.1,
    ):
        super().__init__(
            bert_model_name=bert_model_name,
            label_list=label_list,
            dropout=dropout,
        )
        # Entities for nested spans (no B-/I- prefixes here)
        if type_list is None:
            raise ValueError("Must provide a type_list for nested entities")
        self.type_list = type_list
        # Load BERT config
        config = AutoConfig.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        # Feed-forward nets for start/end representations
        self.start_ffn = nn.Linear(hidden_size, hidden_size)
        self.end_ffn = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        # Biaffine scorer: one logit per entity type
        num_types = len(self.type_list)
        self.biaffine = Biaffine(hidden_size, hidden_size, num_types)
        logger.info(f"Initialized NestedLawTagger with {num_types} entity types")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = self.dropout(outputs.last_hidden_state)
        # Boundary representations
        start_repr = torch.relu(self.start_ffn(seq_output))
        end_repr = torch.relu(self.end_ffn(seq_output))
        start_repr = self.dropout2(start_repr)
        end_repr = self.dropout2(end_repr)
        # Span scores: (batch, seq_len, seq_len, num_types)
        span_scores = self.biaffine(start_repr, end_repr)
        return span_scores
