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
