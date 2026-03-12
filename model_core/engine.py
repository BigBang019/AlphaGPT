import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.distributions import Categorical
from tqdm import tqdm
import json

from .config import ModelConfig
from .data_loader import CryptoDataLoader
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import BTCBacktest


class AlphaEngine:
    """
    AlphaGPT training engine with proper train/val/test evaluation.

    Training loop:
      - RL reward is computed ONLY on training data
      - Validation score is evaluated every N steps
      - Best model (by validation score) is saved
      - Final evaluation is done on test data
    """

    def __init__(
        self,
        use_lord_regularization=True,
        lord_decay_rate=1e-3,
        lord_num_iterations=5,
    ):
        self.loader = CryptoDataLoader()
        self.loader.load_data()

        self.model = AlphaGPT().to(ModelConfig.DEVICE)

        # Optimizer
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        # Low-Rank Decay regularizer
        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"],
            )
            self.rank_monitor = StableRankMonitor(
                self.model, target_keywords=["q_proj", "k_proj"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None

        self.vm = StackVM()
        self.bt = BTCBacktest()

        # Tracking
        self.best_train_score = -float("inf")
        self.best_val_score = -float("inf")
        self.best_formula = None
        self.best_model_state = None

        self.training_history = {
            "step": [],
            "avg_reward": [],
            "best_train_score": [],
            "val_score": [],
            "best_val_score": [],
            "stable_rank": [],
        }

    def _evaluate_formula(self, formula, feat_tensor, raw_data, target_ret):
        """Evaluate a single formula on given data split."""
        res = self.vm.execute(formula, feat_tensor)

        if res is None:
            return -5.0, 0.0

        if res.std() < 1e-4:
            return -2.0, 0.0

        score, ret_val = self.bt.evaluate(res, raw_data, target_ret)
        return score.item(), ret_val

    def _evaluate_batch_on_split(self, seqs, feat_tensor, raw_data, target_ret):
        """Evaluate a batch of formulas on a specific data split."""
        bs = seqs.shape[0]
        rewards = torch.zeros(bs, device=ModelConfig.DEVICE)

        for i in range(bs):
            formula = seqs[i].tolist()
            score, _ = self._evaluate_formula(formula, feat_tensor, raw_data, target_ret)
            rewards[i] = score

        return rewards

    def _validate(self, seqs):
        """
        Run validation: evaluate formulas on validation data.
        Returns (best_val_score, best_formula) from this batch.
        """
        bs = seqs.shape[0]
        best_score = -float("inf")
        best_formula = None

        for i in range(bs):
            formula = seqs[i].tolist()
            score, ret_val = self._evaluate_formula(
                formula,
                self.loader.val_feat,
                self.loader.val_raw,
                self.loader.val_target,
            )
            if score > best_score:
                best_score = score
                best_formula = formula

        return best_score, best_formula

    def train(self):
        lord_status = "with LoRD" if self.use_lord else "without LoRD"
        print(f"🚀 Starting BTC Alpha Mining ({lord_status})...")
        print(
            f"   Train: {self.loader.train_feat.shape[2]} bars | "
            f"Val: {self.loader.val_feat.shape[2]} bars | "
            f"Test: {self.loader.test_feat.shape[2]} bars"
        )

        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))

        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)

            log_probs = []
            tokens_list = []

            # Generate formula sequences
            for _ in range(ModelConfig.MAX_FORMULA_LEN):
                logits, _, _ = self.model(inp)
                dist = Categorical(logits=logits)
                action = dist.sample()

                log_probs.append(dist.log_prob(action))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)

            seqs = torch.stack(tokens_list, dim=1)

            # === Evaluate on TRAINING data only ===
            rewards = self._evaluate_batch_on_split(
                seqs,
                self.loader.train_feat,
                self.loader.train_raw,
                self.loader.train_target,
            )

            # Track best training formula
            best_train_idx = rewards.argmax().item()
            if rewards[best_train_idx].item() > self.best_train_score:
                self.best_train_score = rewards[best_train_idx].item()
                best_train_formula = seqs[best_train_idx].tolist()
                tqdm.write(
                    f"[Train] New best: Score {self.best_train_score:.3f} | "
                    f"Formula {best_train_formula}"
                )

            # Normalize rewards (advantage)
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            # Policy gradient loss
            loss = 0
            for t in range(len(log_probs)):
                loss += -log_probs[t] * adv
            loss = loss.mean()

            # Gradient step
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # Apply Low-Rank Decay
            if self.use_lord:
                self.lord_opt.step()

            # === Validation ===
            avg_reward = rewards.mean().item()
            postfix_dict = {
                "AvgRew": f"{avg_reward:.3f}",
                "BestTrain": f"{self.best_train_score:.3f}",
            }

            if step % ModelConfig.VAL_EVERY_N_STEPS == 0:
                val_score, val_formula = self._validate(seqs)
                postfix_dict["ValScore"] = f"{val_score:.3f}"

                if val_score > self.best_val_score:
                    self.best_val_score = val_score
                    self.best_formula = val_formula
                    self.best_model_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                    tqdm.write(
                        f"[Val] ★ New best: Score {val_score:.3f} | "
                        f"Formula {val_formula}"
                    )

                postfix_dict["BestVal"] = f"{self.best_val_score:.3f}"
                self.training_history["val_score"].append(val_score)
                self.training_history["best_val_score"].append(self.best_val_score)

            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict["Rank"] = f"{stable_rank:.2f}"
                self.training_history["stable_rank"].append(stable_rank)

            self.training_history["step"].append(step)
            self.training_history["avg_reward"].append(avg_reward)
            self.training_history["best_train_score"].append(self.best_train_score)

            pbar.set_postfix(postfix_dict)

        # === Final Test Evaluation ===
        print("\n" + "=" * 60)
        print("📊 Final Evaluation on TEST data (out-of-sample)")
        print("=" * 60)

        if self.best_formula is not None:
            test_score, test_ret = self._evaluate_formula(
                self.best_formula,
                self.loader.test_feat,
                self.loader.test_raw,
                self.loader.test_target,
            )
            print(f"  Best formula (by val): {self.best_formula}")
            print(f"  Validation score:      {self.best_val_score:.4f}")
            print(f"  Test score:            {test_score:.4f}")
            print(f"  Test cumulative ret:   {test_ret:.4%}")
        else:
            print("  No valid formula found during training.")
            test_score = -999

        # Save best formula
        result = {
            "formula": self.best_formula,
            "best_val_score": self.best_val_score,
            "test_score": test_score,
            "train_bars": self.loader.train_feat.shape[2],
            "val_bars": self.loader.val_feat.shape[2],
            "test_bars": self.loader.test_feat.shape[2],
        }
        with open("best_btc_strategy.json", "w") as f:
            json.dump(result, f, indent=2)

        # Save training history
        with open("training_history.json", "w") as f:
            json.dump(self.training_history, f)

        # Save best model weights
        if self.best_model_state is not None:
            torch.save(self.best_model_state, "best_model.pt")
            print("  Model checkpoint saved: best_model.pt")

        print(f"\n✓ Training completed!")
        print(f"  Best val score:   {self.best_val_score:.4f}")
        print(f"  Best formula:     {self.best_formula}")


if __name__ == "__main__":
    eng = AlphaEngine(use_lord_regularization=True)
    eng.train()