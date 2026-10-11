"""HRM-principled text generation via cognition dynamics.

The spatial field destroys discrete byte information (proven: 1.5e-05 MSE
after one step vs 0.019 input). Text flows through the cognition pathway
instead, using HRM-style gated recurrent updates:

  cognition <- gated_update(cognition, byte_embed)
  logits <- cognition @ vocab_head

This preserves the HRM's hierarchical gated recurrence while respecting
that text is sequential-symbolic, not spatiotemporal-field, data.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random


class HRMCognitionText:
    """Text generator using HRM-style cognition dynamics."""

    def __init__(self, vocab_size: int = 257, cog_dim: int = 128, seed: int = 0):
        self.vocab_size = vocab_size
        self.cog_dim = cog_dim
        self.eos_id = 256
        key = random.PRNGKey(seed)
        k1, k2, k3, k4 = random.split(key, 4)
        def xavier(k, shape):
            fan_in = shape[0]
            return random.normal(k, shape, dtype=jnp.float32) * jnp.sqrt(1.0 / fan_in)
        # Byte embedding
        self.embed = xavier(k1, (vocab_size, cog_dim))
        # Gated recurrent cell (HRM-style): input, forget, and candidate gates
        # Input: [cog, embed] (2*cog_dim) -> 3*cog_dim (i, f, g)
        self.W_gates = xavier(k2, (cog_dim * 2, cog_dim * 3))
        self.b_gates = jnp.zeros((cog_dim * 3,), dtype=jnp.float32)
        # Hierarchy: slow context vector (like HRM's hierarchy)
        self.W_hier = xavier(k3, (cog_dim * 2, cog_dim))
        # Output head
        self.W_out = xavier(k4, (cog_dim, vocab_size))

    def _step(self, state, byte_id):
        cog, hier = state
        embed, W_gates, b_gates, W_hier, W_out = (
            self.embed, self.W_gates, self.b_gates, self.W_hier, self.W_out)
        e = embed[byte_id]
        gates = jnp.concatenate([cog, e]) @ W_gates + b_gates
        sig = jax.nn.sigmoid(gates)
        i_gate, f_gate = sig[:self.cog_dim], sig[self.cog_dim:2*self.cog_dim]
        g_cand = jnp.tanh(gates[2*self.cog_dim:])
        new_cog = f_gate * cog + i_gate * g_cand
        # Hierarchy: slow-moving context
        new_hier = 0.95 * hier + 0.05 * (jnp.concatenate([cog, e]) @ W_hier)
        # Combine for output (hierarchy modulates)
        combined = new_cog + 0.3 * new_hier
        logits = combined @ W_out
        return (new_cog, new_hier), logits

    def _forward(self, ids):
        def body(state, bid):
            return self._step(state, bid)
        init = (jnp.zeros((self.cog_dim,), dtype=jnp.float32),
                jnp.zeros((self.cog_dim,), dtype=jnp.float32))
        _, logits = jax.lax.scan(body, init, ids)
        return logits

    def params(self):
        return [self.embed, self.W_gates, self.b_gates, self.W_hier, self.W_out]

    def set_params(self, p):
        self.embed, self.W_gates, self.b_gates, self.W_hier, self.W_out = p

    def train(self, text: str, epochs: int = 200, seq_len: int = 64, lr: float = 0.01, seed: int = 0, batch_size: int = 32):
        data = np.frombuffer(text.encode("utf-8", errors="replace"), dtype=np.uint8).astype(np.int32)
        if len(data) < seq_len + 1:
            raise ValueError(f"Need {seq_len+1} bytes, got {len(data)}")
        key = random.PRNGKey(seed)
        history = []
        for epoch in range(epochs):
            key, skey = random.split(key)
            # Sample a batch of windows
            starts = random.randint(skey, (batch_size,), 0, len(data) - seq_len)
            batch_inp, batch_tgt = [], []
            for s in starts:
                s = int(s)
                window = jnp.asarray(data[s:s + seq_len + 1])
                batch_inp.append(window[:-1])
                batch_tgt.append(window[1:])
            inp = jnp.stack(batch_inp)  # (B, T)
            tgt = jnp.stack(batch_tgt)  # (B, T)
            def loss_fn(p):
                self.set_params(p)
                # vmap over batch
                logits = jax.vmap(self._forward)(inp)  # (B, T, V)
                logp = jax.nn.log_softmax(logits, axis=-1)
                nll = -jnp.take_along_axis(logp, tgt[:, :, None], axis=-1).squeeze(-1)
                return jnp.mean(nll)
            p = self.params()
            lv, grads = jax.value_and_grad(loss_fn)(p)
            # Gradient clipping
            grads = [jnp.clip(g, -1.0, 1.0) for g in grads]
            self.set_params([x - lr * g for x, g in zip(p, grads)])
            history.append(float(lv))
            if (epoch + 1) % 50 == 0:
                print(f"  epoch {epoch+1}: loss={float(lv):.4f}", flush=True)
        return history

    def generate(self, prompt: str = "", max_tokens: int = 200, temperature: float = 0.8, seed: int = 0) -> str:
        key = random.PRNGKey(seed)
        pbytes = prompt.encode("utf-8", errors="replace")
        ids = list(pbytes)
        cog = jnp.zeros((self.cog_dim,), dtype=jnp.float32)
        hier = jnp.zeros((self.cog_dim,), dtype=jnp.float32)
        state = (cog, hier)
        for bid in ids:
            state, _ = self._step(state, int(bid))
        for _ in range(max_tokens):
            prev = ids[-1] if ids else 0
            state, logits = self._step(state, int(prev))
            logits = logits / max(temperature, 1e-6)
            key, skey = random.split(key)
            nid = int(random.categorical(skey, logits))
            if nid == self.eos_id:
                break
            ids.append(nid)
        gen = ids[len(pbytes):] if prompt else ids
        return bytes([i for i in gen if 0 <= i < 256]).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "hrm_cogtext.npz", *[np.asarray(p) for p in self.params()])

    def load(self, path: str | Path) -> None:
        d = np.load(Path(path) / "hrm_cogtext.npz")
        self.set_params([jnp.asarray(d[f"arr_{i}"]) for i in range(5)])
