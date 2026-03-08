"""Pretrain the base model on arithmetic corpus.

The base model only needs to learn the FORMAT of arithmetic (N+M=K),
not necessarily get answers right. This gives the evolutionary loop
a starting point where models can generate syntactically valid output.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from csr_llm.config import get_output_dir, load_config
from csr_llm.model import TinyGPT, create_model, save_model
from csr_llm.task import generate_training_corpus
from csr_llm.tokenizer import encode, load_tokenizer, train_tokenizer

logger = logging.getLogger(__name__)


class ArithmeticDataset(Dataset):
    """Dataset of tokenized arithmetic examples."""

    def __init__(self, examples: list[str], tokenizer, max_len: int = 128):
        self.samples = []
        bos = tokenizer.token_to_id("<bos>")
        eos = tokenizer.token_to_id("<eos>")
        pad = tokenizer.token_to_id("<pad>")

        for ex in examples:
            ids = [bos] + encode(tokenizer, ex) + [eos]
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [pad] * (max_len - len(ids))
            self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s[:-1], s[1:]  # input, target (shifted by 1)


def pretrain(cfg: dict) -> tuple[TinyGPT, Path]:
    """Pretrain base model and return model + checkpoint path."""
    out_dir = get_output_dir(cfg) / "base_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint" / "model.pt"

    device = cfg["experiment"]["device"]
    pcfg = cfg["pretrain"]
    tcfg = cfg["task"]

    # Determine task params
    if "difficulty_schedule" in tcfg:
        first = tcfg["difficulty_schedule"][0]
        ops = first["operations"]
        op_range = first["operand_range"]
    else:
        ops = tcfg["operations"]
        op_range = tcfg["operand_range"]

    # Generate corpus
    logger.info("Generating pretraining corpus...")
    corpus = generate_training_corpus(
        n=pcfg["n_examples"],
        operations=ops,
        operand_range=tuple(op_range),
        seed=cfg["experiment"]["seed"],
    )

    # Train tokenizer
    tok_path = out_dir / "tokenizer.json"
    if tok_path.exists():
        logger.info("Loading existing tokenizer...")
        tokenizer = load_tokenizer(tok_path)
    else:
        logger.info("Training tokenizer...")
        tokenizer = train_tokenizer(corpus, vocab_size=cfg["tokenizer"]["vocab_size"], save_path=tok_path)

    # Create dataset
    dataset = ArithmeticDataset(corpus, tokenizer, max_len=cfg["model"]["max_seq_len"])
    loader = DataLoader(dataset, batch_size=pcfg["batch_size"], shuffle=True, drop_last=True)

    # Create model
    model = create_model(cfg).to(device)
    logger.info(f"Model parameters: {model.n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=pcfg["learning_rate"], weight_decay=pcfg["weight_decay"]
    )

    # Training loop
    log_path = out_dir / "pretrain_log.jsonl"
    log_file = open(log_path, "w")

    model.train()
    global_step = 0
    t0 = time.time()

    for epoch in range(pcfg["n_epochs"]):
        epoch_loss = 0.0
        n_batches = 0

        for input_ids, targets in loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            _, loss = model(input_ids, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            # Warmup
            if global_step <= pcfg["warmup_steps"]:
                lr_scale = global_step / pcfg["warmup_steps"]
                for pg in optimizer.param_groups:
                    pg["lr"] = pcfg["learning_rate"] * lr_scale

        avg_loss = epoch_loss / max(1, n_batches)
        elapsed = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "elapsed_sec": round(elapsed, 1),
            "step": global_step,
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()
        logger.info(f"Epoch {epoch}: loss={avg_loss:.4f} elapsed={elapsed:.1f}s")

        if pcfg.get("save_every_epoch", False):
            epoch_path = out_dir / "checkpoint" / f"model_epoch_{epoch}.pt"
            save_model(model, epoch_path)

    log_file.close()

    # Save final checkpoint
    save_model(model, ckpt_path)
    logger.info(f"Base model saved to {ckpt_path}")

    return model, ckpt_path


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/pilot.yaml")
    pretrain(cfg)
