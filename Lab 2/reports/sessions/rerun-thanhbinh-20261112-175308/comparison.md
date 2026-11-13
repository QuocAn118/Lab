# Thanh Binh source experiment and independent rerun

Completed: 2026-11-12T18:22:42+07:00

Source: Nguyen Thanh Binh, commit 92930e6, original experiment 2026-11-11 on RTX 4060. Rerun: HoangDang, assisted by OpenAI Codex, on RTX 5060. One seed and different PyTorch/CUDA versions; timing is descriptive, not a same-hardware benchmark.

| Configuration | Source val | Rerun val | Change (percentage points) | Source epochs | Rerun epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| resnet18_baseline | 84.80% | 84.76% | -0.04 | 5 | 5 |
| densenet121 | 88.24% | 88.24% | +0.00 | 6 | 6 |
| resnet18_lr1e-2 | 83.48% | 83.52% | +0.04 | 6 | 6 |
| resnet18_bs128 | 85.08% | 85.08% | +0.00 | 6 | 6 |

New selected model: densenet121, epoch 5; test accuracy 87.68%, macro F1 0.8755 on 5,000 samples.
Original selected checkpoint independently rechecked: densenet121; original reported validation 88.24%, current validation 88.24%, current test accuracy 87.68%, macro F1 0.8755. These are new evaluation measurements, not new training or imported historical test results.

Artifacts: [comparison chart](comparison.png), [CSV](comparison.csv), [checkpoint checks](checkpoint_check.json), [TensorBoard checks](tensorboard_check.json), [fresh metrics](reports/metrics.json), [original checkpoint recheck](historical_checkpoint_recheck.json).
