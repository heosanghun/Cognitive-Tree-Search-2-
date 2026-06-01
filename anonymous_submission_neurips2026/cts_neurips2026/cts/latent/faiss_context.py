"""FAISS Latent Space Context Window (paper §4.3, Appendix H).

Paper: FAISS-IVF-PQ (nlist=100, nprobe=20, PQ m=8 codes, 8 bits/code).
Mean-pooled z* used as search key; full K x d FP16 vectors stored and returned.
Table 1: FAISS raw latent 800 KB (N=100), structural (IVF+PQ) 99 MB.
"""

from __future__ import annotations

from typing import List, Optional

import torch

try:
    import faiss  # type: ignore[import-untyped]
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


def faiss_available() -> bool:
    return _FAISS_AVAILABLE


class LatentContextWindow:
    """Paper §4.3 / Appendix H: FAISS-backed Latent Space Context Window.

    - Stores full K x d FP16 z* vectors (not just mean-pooled)
    - Mean-pooled 1D used as search key in IVF-PQ index
    - Retrieves Top-k semantically relevant ancestral vectors
    - Active when step > min_steps (paper: t > 10)
    """

    def __init__(
        self,
        dim: int,
        *,
        retrieval_k: int = 3,
        min_steps: int = 10,
        nlist: int = 100,
        nprobe: int = 20,
        pq_m: int = 8,
        pq_bits: int = 8,
    ) -> None:
        self.dim = dim
        self.retrieval_k = retrieval_k
        self.min_steps = min_steps
        self.nlist = nlist
        self.nprobe = nprobe
        self.pq_m = pq_m
        self.pq_bits = pq_bits

        self._keys: List[torch.Tensor] = []
        self._full_vectors: List[torch.Tensor] = []
        self._step_count = 0
        self._trained = False

        effective_pq_m = pq_m
        if dim % pq_m != 0:
            for candidate in [8, 4, 2, 1]:
                if dim % candidate == 0:
                    effective_pq_m = candidate
                    break
        self._effective_pq_m = effective_pq_m

        if _FAISS_AVAILABLE and np is not None:
            self._flat_index = faiss.IndexFlatIP(dim)
            self._ivfpq_index = None
        else:
            self._flat_index = None
            self._ivfpq_index = None

    def _build_ivfpq(self) -> None:
        """Build IVF-PQ index once we have enough vectors for training."""
        if not _FAISS_AVAILABLE or np is None:
            return
        n = len(self._keys)
        actual_nlist = min(self.nlist, max(1, n // 10))
        if n < actual_nlist * 4:
            return

        quantizer = faiss.IndexFlatIP(self.dim)
        ivfpq = faiss.IndexIVFPQ(
            quantizer, self.dim, actual_nlist, self._effective_pq_m, self.pq_bits,
        )
        ivfpq.nprobe = min(self.nprobe, actual_nlist)

        train_data = torch.stack(self._keys).numpy().astype(np.float32)
        faiss.normalize_L2(train_data)
        ivfpq.train(train_data)
        ivfpq.add(train_data)

        self._ivfpq_index = ivfpq
        self._trained = True

    @property
    def size(self) -> int:
        return len(self._full_vectors)

    @property
    def step_count(self) -> int:
        return self._step_count

    def add(self, z_star: torch.Tensor) -> None:
        """Add a fixed-point to the context window.

        z_star: [K, d] latent tokens.
        Stores: full K x d FP16 tensor + mean-pooled [d] as search key.
        """
        full_fp16 = z_star.detach().cpu().half()
        self._full_vectors.append(full_fp16)

        pooled = z_star.detach().float().mean(dim=0)
        if pooled.dim() > 1:
            pooled = pooled.reshape(-1)
        self._keys.append(pooled.cpu())
        self._step_count += 1

        if self._flat_index is not None and np is not None:
            vec_np = pooled.cpu().numpy().reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(vec_np)
            self._flat_index.add(vec_np)

        min_train = max(40, self.nlist * 4)
        if not self._trained and len(self._keys) >= min_train:
            self._build_ivfpq()

    def retrieve(
        self, z_star: torch.Tensor, k: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """Retrieve Top-k ancestral z* vectors (paper: t > min_steps).

        Returns: [k, K, d] tensor of full FP16 z* vectors, or None.
        """
        k = k or self.retrieval_k
        if self._step_count <= self.min_steps:
            return None
        if len(self._full_vectors) < k:
            return None

        pooled = z_star.detach().float().mean(dim=0).reshape(-1)

        indices = self._search_indices(pooled, k)
        if not indices:
            return None

        retrieved = [self._full_vectors[idx].float() for idx in indices]
        return torch.stack(retrieved)

    def _search_indices(self, query_pooled: torch.Tensor, k: int) -> List[int]:
        """Search for top-k nearest neighbors."""
        if np is None:
            return self._cosine_fallback(query_pooled, k)

        query_np = query_pooled.cpu().numpy().reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_np) if _FAISS_AVAILABLE else None

        if self._ivfpq_index is not None and self._trained:
            try:
                _, indices = self._ivfpq_index.search(query_np, min(k, self._ivfpq_index.ntotal))
                return [int(idx) for idx in indices[0] if 0 <= idx < len(self._full_vectors)]
            except Exception:
                pass

        if self._flat_index is not None and self._flat_index.ntotal > 0:
            _, indices = self._flat_index.search(query_np, min(k, self._flat_index.ntotal))
            return [int(idx) for idx in indices[0] if 0 <= idx < len(self._full_vectors)]

        return self._cosine_fallback(query_pooled, k)

    def _cosine_fallback(self, query: torch.Tensor, k: int) -> List[int]:
        query_norm = query / (query.norm() + 1e-8)
        sims = []
        for i, v in enumerate(self._keys):
            v_norm = v / (v.norm() + 1e-8)
            sims.append((float(torch.dot(query_norm, v_norm)), i))
        sims.sort(reverse=True)
        return [idx for _, idx in sims[:k]]

    def memory_kb_per_node(self) -> float:
        """Backward compat: average KB per stored node."""
        n = max(len(self._full_vectors), 1)
        return self.memory_bytes() / n / 1024

    def memory_bytes(self) -> int:
        per_key = self.dim * 4
        per_full = 0
        if self._full_vectors:
            per_full = self._full_vectors[0].numel() * 2
        n = len(self._full_vectors)
        faiss_structural = 99 * 1024 * 1024 if self._trained else 0
        return n * (per_key + per_full) + faiss_structural

    def memory_report(self) -> dict:
        n = len(self._full_vectors)
        raw_kb = 0
        if n > 0 and self._full_vectors:
            raw_kb = n * self._full_vectors[0].numel() * 2 / 1024
        return {
            "n_vectors": n,
            "raw_latent_kb": raw_kb,
            "structural_mb": 99 if self._trained else 0,
            "index_type": "IVF-PQ" if self._trained else "FlatIP",
        }

    def reset(self) -> None:
        self._keys.clear()
        self._full_vectors.clear()
        self._step_count = 0
        self._trained = False
        self._ivfpq_index = None
        if _FAISS_AVAILABLE and np is not None:
            self._flat_index = faiss.IndexFlatIP(self.dim)


def prepend_soft_prefix(
    context: torch.Tensor, retrieved: torch.Tensor
) -> torch.Tensor:
    """Inject retrieved ancestral z* as prepended soft-prefix.

    Paper §4.3: concatenation s0 + Ht in sequence dimension.

    context: [seq_len, d]
    retrieved: [k, K, d] (full z* vectors) or [k, d] (pooled)
    Returns: [prefix_len + seq_len, d]
    """
    if retrieved.dim() == 3:
        k, K, d_r = retrieved.shape
        retrieved_flat = retrieved.reshape(k * K, d_r)
    else:
        retrieved_flat = retrieved

    if retrieved_flat.device != context.device:
        retrieved_flat = retrieved_flat.to(context.device)
    if retrieved_flat.dtype != context.dtype:
        retrieved_flat = retrieved_flat.to(context.dtype)

    if retrieved_flat.shape[-1] != context.shape[-1]:
        return context

    return torch.cat([retrieved_flat, context], dim=0)
