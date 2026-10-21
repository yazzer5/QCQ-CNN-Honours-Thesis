# v15.5 exploratory figures (not pre-registered)

| Figure | What and why |
|---|---|
| `D0_noise_examples.png` | What the corruption looks like at each sigma and salt-and-pepper p, so the accuracy numbers have a picture. |
| `D1_robustness_curves.png` | How fast each model loses accuracy as the images get noisier, and how much training on noisy images gives back. |
| `D2_transfer_matrix.png` | Whether training at one noise level protects at other levels, and whether the front-end changes that (right panel: F). |
| `D3_head_amplification.png` | Checks whether v15's finding (the QNN head magnifies front-end noise) also holds when the noise is in the pixels. |
| `D4_seed46.png` | Seed 46 was the worst frozen front-end in v13.5, v14 and v15; this shows whether it is also the noise-fragile one here. |
| `D5_T_without46.png` | How much of the training benefit T is the rescue of one bad initialisation (seed 46), at each noise level. |
| `D6_subset_consistency.png` | Whether the primary results point the same way on both 5-class subsets and at 10 classes. |
| `D7_salt_and_pepper.png` | Checks whether the Gaussian-noise ranking of models holds for a different kind of corruption. |
| `D8_training_dynamics.png` | Whether noise augmentation slows convergence or pushes runs towards the 200-epoch cap. |
