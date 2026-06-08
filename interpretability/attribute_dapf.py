import os
import json
import torch
import pandas as pd
import numpy as np
from captum.attr import LayerIntegratedGradients


class DAPFAttributor:
    def __init__(self, prompt_model, tokenizer, device, target_class=1):
        self.prompt_model = prompt_model
        self.tokenizer = tokenizer
        self.device = device
        self.target_class = target_class
        self.prompt_model.eval()

        self.embedding_layer = self.prompt_model.plm.get_input_embeddings()

        self.lig = LayerIntegratedGradients(
            self.forward_func,
            self.embedding_layer
        )

    def forward_func(self, input_ids, attention_mask):
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        # You may need to add other OpenPrompt fields depending on your wrapped batch:
        # token_type_ids, loss_ids, shortenable_ids, etc.
        logits = self.prompt_model(batch)

        if isinstance(logits, dict):
            logits = logits["logits"]

        return logits[:, self.target_class]

    def attribute_batch(self, input_ids, attention_mask, n_steps=32):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        baseline = torch.zeros_like(input_ids).to(self.device)

        attributions, delta = self.lig.attribute(
            inputs=input_ids,
            baselines=baseline,
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
            return_convergence_delta=True
        )

        # Sum over embedding dimensions
        token_scores = attributions.sum(dim=-1)

        return token_scores.detach().cpu(), delta.detach().cpu()

    def decode_attributions(self, input_ids, token_scores):
        rows = []

        for i in range(input_ids.shape[0]):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i].tolist())
            scores = token_scores[i].tolist()

            for position, token in enumerate(tokens):
                rows.append({
                    "sample_index": i,
                    "token_position": position,
                    "token": token,
                    "attribution": scores[position]
                })

        return pd.DataFrame(rows)