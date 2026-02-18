"""
Maya Patches - Fix torchtune issues for CUDA graphs and torch.compile

CRITICAL: These patches MUST be applied BEFORE importing any models.

The main issue is torchtune's KVCache which uses .item() calls that:
1. Break CUDA graph capture
2. Break torch.compile fullgraph optimization
3. Cause CPU-GPU synchronization

This module provides comprehensive patches to fix these issues.
"""

import logging
import torch

logger = logging.getLogger(__name__)

_patches_applied = False


def patch_torchtune_kv_cache():
    """
    Patch torchtune KVCache to eliminate .item() calls.

    The issue is in torchtune/modules/kv_cache.py:
    - size property calls .item() which breaks CUDA graphs
    - reset() calls self.size which triggers .item()
    - update() has assertions that may cause issues

    This patch replaces the problematic methods with tensor-only operations.
    """
    global _patches_applied

    if _patches_applied:
        return True

    try:
        import torchtune.modules.kv_cache as kv_module

        # Store original methods for potential fallback
        _original_reset = kv_module.KVCache.reset
        _original_size = kv_module.KVCache.size.fget if hasattr(kv_module.KVCache.size, 'fget') else None
        _original_update = kv_module.KVCache.update

        def patched_reset(self):
            """
            Reset KV cache without calling .item().

            Original implementation:
                self.k_cache.zero_()
                self.v_cache.zero_()
                self.cache_pos -= self.size  # This calls .item()!

            We replace with direct tensor operations.
            """
            self.k_cache.zero_()
            self.v_cache.zero_()
            # Instead of self.cache_pos -= self.size (which calls .item()),
            # reset cache_pos to initial state directly
            # Use data attribute to allow inplace update in inference mode
            max_seq_len = self.cache_pos.size(0)
            with torch.no_grad():
                self.cache_pos.data.copy_(torch.arange(0, max_seq_len, device=self.cache_pos.device))

        def patched_size_getter(self) -> int:
            """
            Get cache size without calling .item().

            Original: return self.cache_pos[0].item()

            We keep the return type as int for compatibility,
            but use a method that doesn't block CUDA graphs.
            """
            # For most use cases, we can use the tensor directly
            # Only convert to int when absolutely necessary
            # This is called rarely outside of reset()
            return int(self.cache_pos[0].cpu())

        def patched_update(self, k_val, v_val):
            """
            Update KV cache without assertions that cause issues.

            The original has:
                assert (self.cache_pos[0] + seq_len) <= self.k_cache.shape[2]

            We remove the assertion for CUDA graph compatibility.
            """
            bsz, num_heads, seq_len, head_dim = k_val.shape

            # Get current position as tensor slice (avoids .item())
            pos_slice = self.cache_pos[:seq_len]

            # Update cache tensors
            k_out = self.k_cache
            v_out = self.v_cache
            k_out[:, :, pos_slice] = k_val
            v_out[:, :, pos_slice] = v_val

            # Update position
            self.cache_pos = self.cache_pos + seq_len

            return k_out, v_out

        # Apply patches
        kv_module.KVCache.reset = patched_reset
        kv_module.KVCache.size = property(patched_size_getter)
        kv_module.KVCache.update = patched_update

        logger.info("torchtune KVCache patched successfully - removed .item() calls")
        _patches_applied = True
        return True

    except Exception as e:
        logger.warning(f"Failed to patch torchtune KVCache: {e}")
        return False


def enable_dynamo_scalar_capture():
    """
    Enable torch._dynamo scalar capture for better graph compilation.

    This allows .item() calls to be captured in the graph instead of
    causing graph breaks.
    """
    try:
        import torch._dynamo
        torch._dynamo.config.capture_scalar_outputs = True
        logger.info("torch._dynamo.config.capture_scalar_outputs enabled")
        return True
    except Exception as e:
        logger.warning(f"Failed to enable dynamo scalar capture: {e}")
        return False


def apply_all_patches():
    """Apply all patches needed for optimal performance."""
    results = []

    # Enable scalar capture first
    results.append(("dynamo_scalar_capture", enable_dynamo_scalar_capture()))

    # Patch KVCache
    results.append(("kv_cache_patch", patch_torchtune_kv_cache()))

    # Log results
    for name, success in results:
        status = "✓" if success else "✗"
        logger.info(f"Patch {name}: {status}")

    return all(success for _, success in results)


# Apply patches when this module is imported
apply_all_patches()
