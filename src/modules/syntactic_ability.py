import logging
import re
from typing import List

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM


# suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)



class SyntacticAbilityEvaluation:

    def __init__(self, model):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def evaluate_corpus(self, data_df, batch_size):
        results_df = pd.DataFrame(columns=['step', 'sent_id', 'masked_sent', 'label', 'correct', 'probability', 'surprisal', 'description'])
        special_tokens = [re.escape(token) for token in self.tokenizer.all_special_tokens if token != self.tokenizer.mask_token]
        pattern = r'\s*(' + '|'.join(special_tokens) + r')\s*'
        sent_id = 0

        for column in data_df.columns:
            for batch in self.generate_batches(data_df[column].tolist(), batch_size):
                masked_inputs, targets = self.prepare_batch(batch)  # shapes: [num_valid_rows, sequence_length], [num_valid_rows, 2]

                masked_sents = self.tokenizer.batch_decode(masked_inputs)
                masked_sents = [re.sub(pattern, ' ', sentence).strip() for sentence in masked_sents]

                correct_labels = self.tokenizer.convert_ids_to_tokens(targets[:, 0])
                incorrect_labels = self.tokenizer.convert_ids_to_tokens(targets[:, 1])

                steps = list(range(0, 200_000, 20_000)) + list(range(200_000, 2_100_000, 100_000))
                for step in steps:
                    checkpoint = self.model + f'-step_{step//1000}k'
                    probs_correct, surprisals_correct, probs_incorrect, surprisals_incorrect = self.evaluate(masked_inputs, targets, checkpoint)
                    
                    results_df = self.update_df(results_df, 
                                                step, 
                                                sent_id + np.arange(len(masked_sents)), 
                                                masked_sents, 
                                                correct_labels,
                                                probs_correct,
                                                surprisals_correct,
                                                truth_value=True,
                                                description=column)
                    
                    results_df = self.update_df(results_df, 
                                                step, 
                                                sent_id + np.arange(len(masked_sents)), 
                                                masked_sents, 
                                                incorrect_labels,
                                                probs_incorrect,
                                                surprisals_incorrect,
                                                truth_value=False,
                                                description=column)
                    
                sent_id += len(masked_sents)
        
        return results_df.sort_values(['step', 'sent_id']).reset_index(drop=True)
    

    def generate_batches(self, data: List, batch_size: int):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    
    def prepare_batch(self, batch):
        """
        INPUT:
            - batch (List[(str, str)]): a list of sentence pairs
        OUTPUT:
            - masked_inputs: tensor of masked input ids - shape: [num_valid_rows, sequence_length]
            - targets: tensor of target ids for the masked positions - shape: [num_valid_rows, 2]
        """
        correct_sents, incorrect_sents = zip(*batch)

        correct_inputs = self.tokenizer(correct_sents, padding=True, return_tensors='pt').input_ids      # shape: [batch_size, sequence_length]
        incorrect_inputs = self.tokenizer(incorrect_sents, padding=True, return_tensors='pt').input_ids  # shape: [batch_size, sequence_length]

        targets_mask = correct_inputs != incorrect_inputs

        # number of differences in each row
        diff_counts = targets_mask.sum(dim=1)   # shape: [batch_size]

        # only keeping rows with exactly one difference 
        # (removes sentence pairs wherein the targets are split into multiple tokens)
        valid_rows = torch.nonzero(diff_counts == 1).squeeze()  # shape: [num_valid_rows (batch_size - num_invalid_rows)]

        target_indices = torch.nonzero(targets_mask[valid_rows])       # shape: [num_valid_rows, 2]
        targets = torch.zeros(len(valid_rows), 2, dtype=torch.long)    # shape: [num_valid_rows, 2]
        targets[:, 0] = correct_inputs[valid_rows, target_indices[:, 1]]
        targets[:, 1] = incorrect_inputs[valid_rows, target_indices[:, 1]]

        # replacing the target token with the mask token in the sentences
        masked_inputs = correct_inputs[valid_rows].to(self.device)       # shape: [num_valid_rows, sequence_length]
        masked_inputs[torch.arange(masked_inputs.size(0)), target_indices[:, 1]] = self.tokenizer.mask_token_id

        return masked_inputs, targets
    

    def evaluate(self, masked_input_ids, targets, checkpoint):
        """
        - INPUT: 
            masked_input_ids: tensor of masked input ids - shape: [num_valid_rows, sequence_length]
            targets: tensor of target ids for the masked tokens - shape: [num_valid_rows, 2]
            checkpoint: a pretrained model checkpoint
        - OUTPUT: probability and surprisal of the correct and incorrect targets at the masked positions 
        """
        model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(self.device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        attention_mask = (masked_input_ids != self.tokenizer.pad_token_id).long()

        model.eval()
        with torch.no_grad():
            logits = model(input_ids=masked_input_ids, attention_mask=attention_mask).logits    # shape: [num_valid_rows, sequence_length, vocab_size]

        probs = F.softmax(logits, dim=-1)

        # probability and surprisal of the targets at the masked positions
        mask_token_indices = torch.nonzero(masked_input_ids == self.tokenizer.mask_token_id, as_tuple=False)
        batch_indices = torch.arange(probs.size(0), device=self.device)
        
        probs_correct = probs[batch_indices, mask_token_indices[:, 1], targets[:, 0]]    # shape: [num_valid_rows]
        surprisals_correct = -torch.log2(probs_correct)                                  # shape: [num_valid_rows]

        probs_incorrect = probs[batch_indices, mask_token_indices[:, 1], targets[:, 1]]  # shape: [num_valid_rows]
        surprisals_incorrect = -torch.log2(probs_incorrect)                              # shape: [num_valid_rows]

        return probs_correct.cpu().tolist(), surprisals_correct.cpu().tolist(), probs_incorrect.cpu().tolist(), surprisals_incorrect.cpu().tolist()
    
    
    def update_df(self, df: pd.DataFrame, 
                        step, 
                        sent_ids: np.array, 
                        masked_sents: List[str], 
                        labels: List[str], 
                        probabilities: List[float], 
                        surprisals: List[float],
                        truth_value=True, 
                        description=''):
        
        label_df = pd.DataFrame(
                    {'step': [step] * len(sent_ids),
                    'sent_id': sent_ids, 
                    'masked_sent': masked_sents, 
                    'label': labels,
                    'correct': [truth_value] * len(sent_ids), 
                    'probability': probabilities, 
                    'surprisal': surprisals,
                    'description': [description] * len(sent_ids)}
                    )        
        return pd.concat([df, label_df], ignore_index=True)