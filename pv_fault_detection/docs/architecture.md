# Architecture

## System-Level Architecture

See spec Section 5 for the full architecture description.

```
RGB image (224x224x3)
    -> EfficientNet-B0 encoder (ImageNet-pretrained, separate weights from thermal)
    -> GAP -> projection head -> f_rgb (256-d)

Thermal image (224x224x1, + optional raw radiometric temperature channel 224x224x1)
    -> EfficientNet-B0 encoder (separate weights, ImageNet-init)
    -> GAP -> projection head -> f_thermal (256-d)

Electrical sequence:
    I-V curve resampled to 100 points (2 channels: I, V)
    + P-V derived curve (1 channel)
    + scalar features: Voc, Isc, Vmp, Imp, FF, Rs, Rsh (7-d)
    + environmental conditioning: irradiance, ambient temp, wind (3-d)
    -> 1D-CNN (3 conv blocks) -> 2-layer Transformer encoder over curve tokens
    -> f_elec (128-d)

f_rgb, f_thermal -> Gated Cross-Attention block (bidirectional)
    -> fused image feature f_img (256-d)

f_img, f_elec -> concatenation -> MLP bottleneck -> shared representation z (256-d)
    -> [domain-invariance regularization: CORAL / adversarial-DG / domain-norm]
    -> Fault classification head: Linear(256 -> 7), multi-label sigmoid
    -> (training-only) Domain classifier head: Linear(256 -> num_domains), gradient-reversed
```

## Component Table

See spec Section 10 for full component table with backbone details, output dims, and notes.

## Alternatives Considered and Rejected

- **Shared multimodal transformer tokenizer**: rejected — insufficient paired data at current dataset scale.
- **ConvNeXt-Tiny**: available as config-selectable backbone (`model.image_backbone: convnext_tiny`), but not default.
- **Separate per-modality-combination models**: rejected — modality dropout through one model is standard practice and avoids multiplying the experiment matrix.
