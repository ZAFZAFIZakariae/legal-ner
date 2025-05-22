import logging
from typing import List
import torch
from torch import nn
from transformers import AutoModel

logger = logging.getLogger(__name__)

class BaseModel(nn.Module):
    """
    Abstract base for sequence tagging models with checkpointing.
    """
    def __init__(
        self,
        bert_model_name: str = "aubmindlab/bert-base-arabertv2",
        label_list: List[str] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Load pretrained BERT
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)

        # Validate and store labels
        if label_list is None:
            raise ValueError("Must provide a label_list for your tag set")
        self.label_list = label_list
        self.num_labels = len(label_list)
        logger.info(f"Initializing model with {self.num_labels} labels")

    def save(self, path: str):
        """Save model state to the given path."""
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str, device: str = 'cpu'):
        """Load model state from path onto specified device."""
        state = torch.load(path, map_location=device)
        self.load_state_dict(state)
        self.to(device)
        logger.info(f"Model loaded from {path} to {device}")
