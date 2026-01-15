import torch
from typing import Literal

def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    '''
    reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against the ground truths, producing a dict with keys "reward", "format_reward", and "answer_reward".
    rollout_responses: list[str] Rollouts from the policy. The length of this list is rollout_batch_size = n_prompts_per_rollout_batch * group_size.
    repeated_ground_truths: list[str] The ground truths for the examples. The length of this list is rollout_batch_size, because the ground truth for each example is repeated group_size times.
    group_size: int Number of responses per question (group).
    advantage_eps: float Small constant to avoid division by zero in normalization.
    normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise subtract only the group mean.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
            - advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout response.
            - raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout response.
            - metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    '''
    # Compute raw rewards for each response
    raw_rewards = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, ground_truth)
        raw_rewards.append(reward_dict["reward"])
    raw_rewards = torch.tensor(raw_rewards)

    # Reshape rewards into groups
    n_groups = len(raw_rewards) // group_size
    grouped_rewards = raw_rewards.view(n_groups, group_size)

    # Compute group statistics
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    if normalize_by_std:
        group_stds = grouped_rewards.std(dim=1, keepdim=True) + advantage_eps
        advantages = (grouped_rewards - group_means) / group_stds
    else:
        advantages = grouped_rewards - group_means

    # Flatten advantages back to original shape
    advantages = advantages.view(-1)

    # Compute metadata statistics
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
        "mean_advantage": advantages.mean().item(),
        "std_advantage": advantages.std().item(),
    }

    return advantages, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    '''
    raw_rewards_or_advantages: torch.Tensor, shape (batch_size, 1).
    policy_log_probs: torch.Tensor, shape (batch_size, sequence_length).

    Returns:
    torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to be aggregated across the batch and sequence dimensions in the training loop).
    '''
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''
    advantages: torch.Tensor, shape (batch_size, 1).
    policy_log_probs: torch.Tensor, shape (batch_size, sequence_length).
    old_log_probs: torch.Tensor, shape (batch_size, sequence_length).
    cliprange: float, the clip range for the ratio.

    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
    loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped loss.
    metadata dict containing whether each token was clipped or not.
    '''
    # Calculate probability ratio r = π_θ(a|s) / π_θ_old(a|s)
    ratio = torch.exp(policy_log_probs - old_log_probs)  # shape: (batch_size, sequence_length)
    
    # Calculate surrogate objectives
    surr1 = ratio * advantages  # Unclipped surrogate
    surr2 = torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * advantages  # Clipped surrogate
    
    # Take the minimum of the surrogates (negative since we want to maximize reward)
    loss = -torch.min(surr1, surr2)
    
    # Track which tokens were clipped (where surr2 < surr1)
    was_clipped = (surr2 < surr1)
    
    metadata = {
        "clipped_tokens": was_clipped,
        "clip_fraction": was_clipped.float().mean()
    }

    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''
    Select and compute the desired policy-gradient loss.
    policy_log_probs (batch_size, sequence_length)
    raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
    advantages Required for "reinforce_with_baseline" and "grpo_clip"; shape (batch_size, 1).
    old_log_probs Required for "grpo_clip"; shape (batch_size, sequence_length).
    cliprange Required for "grpo_clip"; float.

    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
    loss (batch_size, sequence_length), per-token loss.
    metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    '''
    # Assert input shapes
    assert policy_log_probs.dim() == 2, f"Expected policy_log_probs to have 2 dimensions, got {policy_log_probs.dim()}"
    
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {"loss_type": "no_baseline"}
        
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {"loss_type": "reinforce_with_baseline"}
        
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
        
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    '''
    Compute the mean of tensor along a given dimension, considering only those elements where mask == 1.
    tensor: torch.Tensor The data to be averaged.
    mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
    dim: int | None Dimension over which to average. If None, compute the mean over all masked elements.

    Returns:
    torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    '''
    return torch.sum(tensor * mask, dim=dim) / torch.sum(mask, dim=dim)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''
    Return:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
    loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return this so we can log it.
    metadata Dict with metadata from the underlying loss call, and any other statistics you might want to log.

    You should call loss.backward() in this function. Make sure to adjust for gradient accumulation.
    '''
    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange) # (batch_size, sequence_length)
    loss = masked_mean(loss, response_mask)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss, metadata
