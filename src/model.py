import logging
from typing import List
from torch import nn
from transformers import AutoConfig

from src.base_model import BaseModel

logger = logging.getLogger(__name__)

class LawTagger(BaseModel):
    """
    Sequence tagger for legal entities. Inherits BERT + checkpointing from BaseModel.
    """
    def __init__(
        self,
        bert_model_name: str = "aubmindlab/bert-base-arabertv2",
        label_list: List[str] = None,
        dropout: float = 0.1,
    ):
        # Initialize BaseModel (loads BERT, dropout, label_list)
        super().__init__(
            bert_model_name=bert_model_name,
            label_list=label_list,
            dropout=dropout,
        )
        # Configure classifier head
        config = AutoConfig.from_pretrained(bert_model_name, num_labels=self.num_labels)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        logger.info("Classifier head initialized")

    def forward(self, input_ids, attention_mask):
        # BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        # Token-level logits
        logits = self.classifier(sequence_output)
        return logits
