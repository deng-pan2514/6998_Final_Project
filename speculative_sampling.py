import torch
from tqdm import tqdm
import torch

from kv_cache import KVCacheModel
from utils import norm_logits, sample, max_fn
from decoder import Decoder

@torch.no_grad()
def speculative_sampling(input_seq: torch.Tensor, small_model: torch.nn.Module, large_model: torch.nn.Module,
                                  max_sequence_length: int, num_guesses: int = 4,
                                  temp: float = 1.0, top_k_value: int = 0, top_p_value: float = 0.0, debug_mode: bool = False, seed_val: int = None) -> torch.Tensor:
    """
    speculative sampling algorithm with KV Cache Optimization.
    Based on the research paper: https://arxiv.org/pdf/2211.17192.pdf

    Args:
        input_seq (torch.Tensor): Initial input sequence, shape (1, initial_seqlen).
        small_model (torch.nn.Module): The smaller, approximation model.
        large_model (torch.nn.Module): The larger, target model.
        max_sequence_length (int): Maximum length of the generated sequence.
        num_guesses (int): Number of tokens the small model predicts in parallel.
        temp (float): Temperature for sampling, default is 1.0.
        top_k_value (int): Optional, for top-k sampling, default is 0.
        top_p_value (float): Optional, for top-p (nucleus) sampling, default is 0.0.
        debug_mode (bool): Flag to print debug information, default is False.
        seed_val (int): Optional, for setting a random seed for reproducibility.

    Returns:
        torch.Tensor: The generated sequence of tokens (1, generated_seqlen).
    """
    current_length = input_seq.shape[1]
    total_length = current_length + max_sequence_length

    assert input_seq.shape[0] == 1, "Batch size must be 1 for this implementation"
    assert small_model.device == large_model.device, "Both models must be on the same device"

    device = large_model.device
    small_cache = KVCacheModel(small_model, temp, top_k_value, top_p_value)
    large_cache = KVCacheModel(large_model, temp, top_k_value, top_p_value)

    rejections = 0
    large_samples = 0
    accepted_tokens = 0

    while input_seq.shape[1] < total_length:
        # Generate guesses using the small model
        current_prefix_length = input_seq.shape[1]
        guesses = small_cache.generate(input_seq, num_guesses)
        large_cache.generate(guesses, 1)
        
        num_tokens_considered = current_prefix_length + num_guesses - 1

        for idx in range(num_guesses):
            if seed_val:
                torch.manual_seed(seed_val)
            random_number = torch.rand(1, device=device)
            current_token = guesses[:, current_prefix_length + idx]

            if random_number > (large_cache._prob_history[:, current_prefix_length + idx - 1, current_token]) / (small_cache._prob_history[:, current_prefix_length + idx - 1, current_token]):
                # Reject the guess
                num_tokens_considered = current_prefix_length + idx - 1
                break

            if debug_mode:
                print(f"Accepted guess: {current_token[0]} - {Decoder().decode(torch.tensor([current_token]))}")

            accepted_tokens += 1

        assert num_tokens_considered >= current_prefix_length - 1, f"Error in token count: {num_tokens_considered}, {current_prefix_length}"

        input_seq = guesses[:, :num_tokens_considered + 1]
        small_cache.rollback(num_tokens_considered + 1)

        if num_tokens_considered < current_prefix_length + num_guesses - 1:
            # Resample from the large model
            next_token = sample(max_fn(large_cache._prob_history[:, num_tokens_considered, :] - small_cache._prob_history[:, num_tokens_considered, :]))
            if debug_mode:
                print(f"Resample at position {num_tokens_considered}: {Decoder().decode(next_token)}")
            rejections += 1
            large_cache.rollback(num_tokens_considered + 1)
        else:
            # All guesses accepted
            assert num_tokens_considered == large_cache._prob_history.shape[1] - 1
            next_token = sample(large_cache._prob_history[:, -1, :])
            if debug_mode:
                print(f"Sampled next token: {Decoder().decode(next_token)}")
            large_samples += 1
            large_cache.rollback(num_tokens_considered + 2)

        input_seq = torch.cat((input_seq, next_token), dim=1)

    if debug_mode:
        print(f"Total generated tokens: {input_seq.shape[-1] - current_length}, accepted: {accepted_tokens}, large model samples: {large_samples}, rejections: {rejections}")
    return input_seq
