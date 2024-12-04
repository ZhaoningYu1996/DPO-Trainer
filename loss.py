import torch
import torch.nn as nn
import torch.nn.functional as F

def dpo_loss(model: nn.Module, reference_model: nn.Module, inputs: dict, beta: float, device: torch.device) -> torch.Tensor:
    """
    Computes the Direct Preference Optimization loss for a batch of inputs

    Args:
        model: The current policy model.
        reference_model: The reference policy model.
        inputs: A batch of inputs from the DataLoader, containing:
            - 'instruction_ids': Tensor of input token IDs.
            - 'chosen_ids': Tensor of chosen output token IDs.
            - 'rejected_ids': Tensor of rejected output token IDs.
        beta: The temperature controlling strength of KL penalty.
        device: The device to perform computations on.

    Returns:
        loss: The computed DPO loss.
    """
    # Extract input tensors and move them to the specified device
    instruction_ids = inputs['instruction_ids'].to(device)
    chosen_ids = inputs['chosen_ids'].to(device)
    rejected_ids = inputs['rejected_ids'].to(device)

    # Concatenate instruction_ids with chosen_ids and rejected_ids to get the full sequences for model input
    chosen_input = torch.cat([instruction_ids, chosen_ids], dim=1)
    rejected_input = torch.cat([instruction_ids, rejected_ids], dim=1)

    # Get model outputs for both chosen and rejected inputs
    chosen_outputs = model(chosen_input)
    rejected_outputs = model(rejected_input)

    # Get reference model outputs for both chosen and rejected inputs
    with torch.no_grad():
        ref_chosen_outputs = reference_model(chosen_input)
        ref_rejected_outputs = reference_model(rejected_input)

    # Extract logits from the model outputs and reference model outputs
    chosen_logits = chosen_outputs.logits
    rejected_logits = rejected_outputs.logits
    ref_chosen_logits = ref_chosen_outputs.logits
    ref_rejected_logits = ref_rejected_outputs.logits

    # Compute log probabilities using log_softmax
    chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
    rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)
    ref_chosen_log_probs = F.log_softmax(ref_chosen_logits, dim=-1)
    ref_rejected_log_probs = F.log_softmax(ref_rejected_logits, dim=-1)

    # Shift the sequences to get the targets
    chosen_targets = chosen_input[:, 1:]
    rejected_targets = rejected_input[:, 1:]

    # Adjust log_probs to align with the targets
    chosen_log_probs = chosen_log_probs[:, :-1, :]
    rejected_log_probs = rejected_log_probs[:, :-1, :]
    ref_chosen_log_probs = ref_chosen_log_probs[:, :-1, :]
    ref_rejected_log_probs = ref_rejected_log_probs[:, :-1, :]

    # Gather the log probabilities corresponding to the target tokens
    chosen_token_log_probs = chosen_log_probs.gather(2, chosen_targets.unsqueeze(-1)).squeeze(-1)
    rejected_token_log_probs = rejected_log_probs.gather(2, rejected_targets.unsqueeze(-1)).squeeze(-1)

    ref_chosen_token_log_probs = ref_chosen_log_probs.gather(2, chosen_targets.unsqueeze(-1)).squeeze(-1)
    ref_rejected_token_log_probs = ref_rejected_log_probs.gather(2, rejected_targets.unsqueeze(-1)).squeeze(-1)

    # Get the length of the prompt sequence
    input_length = instruction_ids.size(1)

    # Create masks to exclude the prompt tokens from loss computation, we only care about the response tokens
    chosen_mask = torch.ones_like(chosen_token_log_probs)
    rejected_mask = torch.ones_like(rejected_token_log_probs)

    # Zero out positions corresponding to the prompt tokens
    chosen_mask[:, :input_length - 1] = 0
    rejected_mask[:, :input_length - 1] = 0

    # Apply masks to the token log probabilities
    chosen_token_log_probs = chosen_token_log_probs * chosen_mask
    rejected_token_log_probs = rejected_token_log_probs * rejected_mask
    ref_chosen_token_log_probs = ref_chosen_token_log_probs * chosen_mask
    ref_rejected_token_log_probs = ref_rejected_token_log_probs * rejected_mask

    # Sum the log probabilities over the response tokens
    chosen_log_prob = chosen_token_log_probs.sum(dim=1)
    rejected_log_prob = rejected_token_log_probs.sum(dim=1)
    ref_chosen_log_prob = ref_chosen_token_log_probs.sum(dim=1)
    ref_rejected_log_prob = ref_rejected_token_log_probs.sum(dim=1)

    # Compute the difference in log probabilities adjusted by the reference model
    log_prob_diff = (chosen_log_prob - rejected_log_prob) - (ref_chosen_log_prob - ref_rejected_log_prob)

    # Compute the DPO loss using the logistic loss function
    loss = -F.logsigmoid(beta * log_prob_diff).mean()

    return loss
