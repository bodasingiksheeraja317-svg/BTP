# Domain Generalization

## Methods Compared (spec Section 12)

| Method | File | Mechanism | Extra loss? |
|---|---|---|---|
| ERM baseline | n/a | Plain pooled training | None |
| CORAL | `coral.py` | Covariance alignment in-batch | coral_weight * coral_loss |
| Adversarial DG | `adversarial_dg.py` | Gradient reversal + domain classifier | adversarial_weight * domain_loss |
| Domain-specific norm | `domain_norm.py` | Per-domain BatchNorm in bottleneck | None (architectural only) |
| MMD | `coral.py` | MMD alignment | mmd_weight * mmd_loss |

All methods attach at shared bottleneck `z` (256-d), never inside modality encoders.

## LODO Evaluation

DG methods are evaluated EXCLUSIVELY via LODO protocol.
The config loader rejects `dg_method != none` with `protocol: random_split` at load time.

## Adding a New DG Method

1. Create `src/models/domain_generalization/<method>.py`.
2. Add the method's loss/transformation to `pv_fault_model.py` (attach at `z`, never in encoders).
3. Add corresponding loss term to `trainer.py`'s loss assembly.
4. Add `model.dg_method: <method>` to the config and ablation matrix.
5. Add `<method>_weight` config key to `configs/base.yaml`.
6. Ensure `evaluation.protocol: lodo` is enforced in `config.py`'s `_validate()`.

## Future Work (out of scope for current implementation)

- Self-supervised pretraining on massive unlabeled imagery: explicitly out of scope
  per source blueprint. Future work direction if data scale grows.
- Test-time training with gradient updates: beyond BatchNorm recalibration (TTA mode).
