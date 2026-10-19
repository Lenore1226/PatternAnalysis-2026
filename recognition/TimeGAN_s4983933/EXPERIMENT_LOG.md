# TimeGAN Experiment Log

## V4.0–V4.5: Model Improvement Experiments

### V4.0 — Mid-price Return Representation
Added mid-price log returns as the 41st feature.

- Return KL: 3.180144
- Generated return standard deviation: 9.38e-6
- Finding: The model can generate returns, but output diversity remains limited.

### V4.1 — Asinh and Z-score Normalisation
Replaced Min-Max scaling of the return feature with asinh transformation and Z-score normalisation.

- Return KL: 2.133785
- Generated return standard deviation: 1.28e-7
- Finding: Histogram KL improved, but return diversity deteriorated.

### V4.2 — Identity Recovery Activation
Changed the Recovery output activation from Sigmoid to Identity.

- Return KL: 2.133785
- Generated return standard deviation: 1.24e-7
- Finding: Training-level noise sensitivity improved, but generated returns still exhibited severe collapse.

### V4.3 — Weighted Return Reconstruction
Added a reconstruction objective that gives higher weight to nonzero mid-price returns.

- Validation reconstruction correlation: 0.9750
- Generated return KL: 3.762186
- Generated return standard deviation: 3.61e-6
- Finding: Return reconstruction improved substantially, but generated returns did not reproduce the real market's zero-return frequency and volatility.

### V4.4 — Independent Empirical Tick Baseline
Sampled discrete mid-price movements from the training-set empirical distribution.

- Return KL: 0.006692
- Lag-1 absolute-return correlation: -0.0026
- Finding: Marginal distribution matching improved, but the independent baseline failed to reproduce temporal dependence.

### V4.5 — Markov Tick Baseline
Modelled the probability of a price movement conditional on whether the previous observation contained a movement.

- Return KL: 0.004997
- Lag-1 absolute-return correlation: 0.0432
- Real test lag-1 absolute-return correlation: 0.2350
- Finding: First-order movement dependence improved temporal structure, but did not fully capture volatility clustering.

## Key Research Findings

1. Approximately 89% of training-set observations have zero mid-price returns.
2. All observed nonzero changes in the raw doubled mid-price were integer multiples of 100.
3. Weighted reconstruction losses significantly improved the model's ability to reconstruct sparse financial returns.
4. Matching marginal return distributions does not guarantee realistic time dependence.
5. Statistical tick baselines are not substitutes for a complete TimeGAN-based limit order book generator.

## Next Research Direction — V5.0

Investigate a tick-aware TimeGAN with explicit price-movement modelling, while maintaining consistency between generated mid-price paths and the ten-level order book.

## 2026-10-19 — V5.1–V5.3 Constraint-aware LOB Experiments

### Objective
Investigate whether a constraint-aware LOB decoder can improve market-structure validity while preserving reconstruction quality and generative diversity.

### V5.1: Structural parameterization
- Verified positive bid-ask spreads and ordered ask/bid levels.
- Represented 20 relative price features using one spread and 18 level gaps.
- Maximum price reconstruction error: 2.84e-14.
- Structural parameterization test passed.

### V5.2: Constraint-aware LOB Head
- Implemented a differentiable PyTorch LOB head.
- Verified Min-Max normalization and gradient propagation.
- Frozen-embedder test normalized LOB MSE: 0.184861.
- Original V4.3 Recovery test normalized LOB MSE: 0.011923.
- Found excessively large initial spread and gap values.

### V5.3: Mean-informed initialization
- Initialized output biases using training-set statistics only.
- Training-set mean spread: 0.139864.
- Mean ask gap: 0.019693.
- Mean bid gap: 0.019951.
- Test normalized LOB MSE: 0.010218.
- Original V4.3 Recovery test normalized LOB MSE: 0.011923.
- Test spread MAE: 0.051112 versus 0.051315 for V4.3.
- All tested market-structure constraints passed.
- Initial validation MSE was already lower than the best trained-epoch MSE, suggesting initialization was the primary improvement.

### Generative evaluation
- Reused the frozen V4.3 Generator and Supervisor with the V5.3 LOB Head.
- Noise distribution: uniform.
- Generated 512 sequences of length 24.
- Real spread standard deviation: 0.041816.
- Generated spread standard deviation: 0.000158.
- Real pairwise LOB difference: 0.396822.
- Generated pairwise LOB difference: 0.0000128.
- Generated market-structure constraints passed.
- Severe lack of generative diversity remained.

### Conclusion
Mean-informed initialization substantially improved constrained LOB reconstruction. However, strong reconstruction results did not translate into realistic generative diversity. The constraint-aware decoder ensures structural validity but does not by itself resolve TimeGAN mode collapse.

These experiments demonstrate the distinction between reconstruction quality, market validity, and generative realism.

### Project decision
With the submission deadline on 2026-10-30, stop expanding the model architecture. Preserve V3.3 as the stable baseline and document V4.3 and V5.3 as experimental investigations. Prioritize final evaluation, reproducibility, documentation, and submission.
