import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    # Tokenize prompts and outputs separately
    prompt_tokens = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompt_strs]
    output_tokens = [tokenizer.encode(output, add_special_tokens=False) for output in output_strs]
    
    # Concatenate prompt and output tokens
    prompt_and_output_tokens = [p + o for p, o in zip(prompt_tokens, output_tokens)]
    prompt_and_output_lens = [len(tokens) for tokens in prompt_and_output_tokens]
    max_len = max(prompt_and_output_lens)
    
    # Pad sequences to max length
    padded_tokens = [
        tokens + [tokenizer.pad_token_id] * (max_len - len(tokens))
        for tokens in prompt_and_output_tokens
    ]

    # Convert to tensors
    input_ids = torch.tensor(padded_tokens)
    labels = input_ids[:, 1:]  
    input_ids = input_ids[:, :-1]
    
    # Create response mask
    response_mask = torch.zeros_like(input_ids)
    for i, (p_len, o_len) in enumerate(zip([len(p) for p in prompt_tokens], [len(o) for o in output_tokens])):
        # Set mask to 1 for output tokens (after prompt)
        response_mask[i, (p_len-1):(p_len + o_len - 1)] = 1
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }
    
def compute_entropy(logits):
    # Compute the entropy of the logits using log_softmax for numerical stability
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def get_response_log_probs(model, input_ids, labels, return_token_entropy=False, no_grad=True):
    if no_grad:
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            logits = outputs.logits # (batch_size, seq_len, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1) # (batch_size, seq_len, vocab_size)
            # Get log probs of the actual label tokens
            batch_size, seq_len = labels.shape # (batch_size, seq_len)
            log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            if return_token_entropy:
                entropy = compute_entropy(logits)
            else:
                entropy = None
    else:
        outputs = model(input_ids, labels=labels)
        logits = outputs.logits # (batch_size, seq_len, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1) # (batch_size, seq_len, vocab_size)
        # Get log probs of the actual label tokens
        batch_size, seq_len = labels.shape # (batch_size, seq_len)
        log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        if return_token_entropy:
            entropy = compute_entropy(logits)
        else:
            entropy = None
            
    return {
        "log_probs": log_probs,
        "token_entropy": entropy
    }

def masked_normalize(tensor, mask, normalize_constant, dim):
    # Apply mask to tensor (set masked elements to 0)
    masked_tensor = tensor * mask
    
    # Sum along specified dimension
    sum_tensor = torch.sum(masked_tensor, dim=dim)
    
    # Normalize by constant
    normalized = sum_tensor / normalize_constant
    
    return normalized 

def sft_microbatch_train_step(
    policy_log_probs,
    response_mask,
    gradient_accumulation_steps,
    normalize_constant = 1.0
):
    # Compute negative log likelihood loss
    # Note: policy_log_probs are already log probabilities, so we just need to negate them
    nll = -policy_log_probs
    
    # Normalize the loss using the response mask
    # This ensures we only consider loss for response tokens and properly average
    loss = masked_normalize(
        nll,
        response_mask,
        normalize_constant=normalize_constant,
        dim=-1  # Sum over sequence length
    )
    
    # Scale loss by gradient accumulation steps
    loss = loss / gradient_accumulation_steps
    
    # Take mean across batch to get scalar loss
    loss = loss.mean()
    
    # Now backward() will work since loss is a scalar
    loss.backward()

    metadata = {
        "per_sample_loss": loss.item()  # Store the scalar loss value
    }
    
    return loss, metadata
