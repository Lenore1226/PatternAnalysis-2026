# TimeGAN for Synthetic Limit Order Book Generation

**Course:** COMP3710 / COMP7710 — Pattern Recognition  
**Project:** Project 7 — Time-Series Generation (Hard Difficulty)  
**Student:** Meiling Si (s4983933)  
**Dataset:** LOBSTER AMZN Level 10, 21 June 2012  
**Framework:** PyTorch  
**Status:** Research prototype

## 1. Introduction

Financial limit order books (LOBs) record the prices and quantities of outstanding buy and sell orders at different price levels. They provide information about market liquidity, trading activity, and short-term price dynamics.

Realistic synthetic LOB data can support the development and testing of algorithmic trading strategies without relying entirely on expensive historical market datasets. However, generating realistic order books is more challenging than producing ordinary time-series data.

A valid order book must satisfy fundamental market microstructure constraints, including positive bid–ask spreads, correctly ordered price levels, and nonnegative order quantities. Synthetic sequences should also reproduce realistic distributions and temporal variability.

This project investigates whether TimeGAN can generate realistic LOB sequences using historical LOBSTER data.

The main engineering question is:

**Can TimeGAN learn realistic temporal patterns while maintaining valid market microstructure, and does its additional complexity offer meaningful advantages over simple baseline generators?**

The experiments suggest that TimeGAN can generate order books satisfying many basic structural constraints, but the final model suffers from insufficient sample diversity and poor distributional matching.

Consequently, the current implementation is considered a research prototype rather than a reliable simulator for trading strategy backtesting.

## 2. Feasibility Review

### 2.1 User Need and Project Scope

The intended users are quantitative researchers and financial engineers developing market simulation and risk-analysis systems.

The prototype aims to generate synthetic sequences of ten-level limit order books based on LOBSTER data.

The principal engineering challenge is balancing distributional fidelity, temporal realism, market validity, and computational cost.

The initial acceptance criteria are:

1. Generate complete ten-level order books with positive bid–ask spreads, correctly ordered price ladders, and nonnegative quantities.
2. Evaluate spread and mid-price return distributions using KL divergence, with a target of KL ≤ 0.1.
3. Evaluate structural similarity between real and synthetic order-book depth heatmaps, targeting SSIM > 0.6.
4. Compare the proposed model against baseline methods and investigate 3–5 representative failure cases.
5. Report computational requirements, including training time, memory consumption, and generation performance.

### 2.2 Model Choice and Course Concepts

The selected model is TimeGAN, a generative adversarial network designed for sequential data.

TimeGAN combines adversarial training with supervised temporal learning to encourage generated sequences to reproduce realistic time-dependent patterns.

The architecture contains five components:

- Embedder
- Recovery network
- Generator
- Supervisor
- Discriminator

The project draws upon recurrent neural networks, generative adversarial learning, latent representation learning, supervised temporal prediction, and statistical distribution comparison.

The chosen dataset is the LOBSTER AMZN Level 10 sample, containing ten buy-side and ten sell-side price and quantity levels.

This is classified as a Hard Difficulty project.

### 2.3 Preliminary Feasibility Evidence

The initial data audit identified 269,748 aligned order-book records with 40 features per record.

A chronological split was adopted to prevent future information from leaking into model training.

Each sample was divided into sequences of 24 time steps.

An initial simplified TimeGAN smoke test was successfully executed on an NVIDIA A100 GPU using a smaller three-feature representation.

The test confirmed that the model components could execute and generate finite outputs, although it did not demonstrate convergence or realistic order-book generation.

The preliminary GPU smoke test required approximately 3.96 seconds and recorded 27.15 MB of allocated GPU memory.

These measurements refer only to the preliminary simplified experiment, not the final 40-feature model.

### 2.4 Risks, Computational Budget, and Fallback

The main anticipated risks included:

- Adversarial training instability and mode collapse.
- Violations of financial market microstructure constraints.
- Distribution differences between earlier and later trading periods.
- Computational expense.
- Limited generalisation from a single trading-day sample.

The initial plan was to begin with a small working model, verify the data processing and loss functions, and then expand to complete ten-level order-book generation.

If stable generative training could not be achieved, simple baseline methods would be retained to demonstrate the limitations of the proposed model and establish a meaningful comparison.

## 3. Dataset and Preprocessing

### 3.1 Dataset Description

The project uses the LOBSTER AMZN Level 10 sample from 21 June 2012.

Each order-book snapshot contains 40 numerical features.

For each of ten depth levels, the data contains:

| Feature | Description |
|---|---|
| Ask Price | Selling price at the corresponding level |
| Ask Size | Available sell-side quantity |
| Bid Price | Buying price at the corresponding level |
| Bid Size | Available buy-side quantity |

The original dataset contains 269,748 order-book records.

### 3.2 Chronological Data Splitting

To prevent look-ahead bias, the dataset was divided chronologically rather than randomly.

| Split | Number of sequences |
|---|---:|
| Training | 7,867 |
| Validation | 1,685 |
| Testing | 1,685 |

Each sequence contains 24 time steps and 40 features.

The sequence stride was set to 24, producing non-overlapping windows.

### 3.3 Relative Mid-Price Representation

The initial implementation used absolute price values.

However, analysis identified substantial differences in absolute price levels between the training and testing periods.

To reduce this representation problem, V2 preprocessing converts prices into offsets from the contemporaneous mid-price.

The mid-price is defined as:

\[
M_t = \frac{P^{ask}_{1,t}+P^{bid}_{1,t}}{2}
\]

Each price feature is transformed as:

\[
P'_{i,t}=P_{i,t}-M_t
\]

This representation preserves within-snapshot price differences, including bid–ask spreads and price-level ordering.

However, it removes the absolute mid-price trajectory.

As a result, the final V2 model cannot independently generate mid-price returns.

### 3.4 Volume Transformation and Normalisation

Order quantities are transformed using:

\[
V'=\log(1+V)
\]

This reduces the influence of unusually large volume values.

Min–max normalisation is subsequently applied using parameters fitted exclusively on the training partition.

The same normalisation parameters are applied to the validation and testing partitions to avoid data leakage.

Generated values can be inverse-transformed to recover relative price offsets and order quantities.

## 4. TimeGAN Architecture

The implementation uses five GRU-based neural networks.

### 4.1 Embedder

The Embedder transforms real order-book sequences into a lower-dimensional latent representation.

### 4.2 Recovery Network

The Recovery network reconstructs the original feature representation from the latent embedding.

Together, these components learn a representation of the real data.

### 4.3 Generator

The Generator maps random noise sequences to synthetic latent representations.

### 4.4 Supervisor

The Supervisor learns temporal relationships between successive latent representations.

This component encourages generated sequences to reproduce sequential dependencies.

### 4.5 Discriminator

The Discriminator attempts to distinguish real latent representations from synthetic representations.

Adversarial optimisation trains the Generator to produce representations that are difficult to distinguish from real data.

## 5. Training Procedure

TimeGAN is trained in three stages.

### Stage 1: Embedding Pretraining

The Embedder and Recovery network are optimised using reconstruction loss.

### Stage 2: Supervisor Pretraining

The Supervisor is trained to predict successive latent representations using supervised temporal loss.

### Stage 3: Joint Adversarial Training

The complete architecture is optimised using adversarial, supervised, reconstruction, and moment-matching objectives.

Separate parameter updates are performed for the Generator/Supervisor, Embedder/Recovery, and Discriminator.

### 5.1 Final Experiment Configuration

| Parameter | Value |
|---|---|
| Dataset representation | V2 relative mid-price |
| Model architecture | TimeGAN V3.2 |
| Generator output activation | Identity |
| Hidden dimension | 24 |
| Noise dimension | 40 |
| GRU layers | 2 |
| Batch size | 64 |
| Embedding epochs | 2 |
| Supervisor epochs | 2 |
| Joint training epochs | 10 |
| Generator updates per batch | 2 |
| Joint learning rate | 0.0001 |
| Training device | CPU |
| Total runtime | 460.38 seconds |

### 5.2 Model Development Experiments

| Version | Main modification | Observation |
|---|---|---|
| V1 | Absolute-price representation | Strong temporal distribution shift |
| V2 | Relative mid-price representation | Reduced representation shift; mode collapse remained |
| V3 | Gradient isolation during joint training | No meaningful diversity improvement |
| V3.1 | Two generator updates per batch | Mode collapse persisted |
| V3.2 | Identity activation in Generator | Initial diversity followed by rapid collapse |
| V3.3 | Reduced joint learning rate | More stable early training |
| V3.3 (10 epochs) | Extended joint training | Increased sample variation but adversarial imbalance persisted |

### 5.3 Training Stability

During the final ten-epoch experiment, noise sensitivity increased from approximately 0.0000121 to 0.0002396.

Sample standard deviation increased from 0.00001102 to 0.00023998.

However, the Discriminator loss decreased from 2.0030 to 0.2136, while the Generator adversarial loss increased from 1.5016 to 6.8696.

These results suggest that the Generator became more responsive to random noise, but the Discriminator increasingly distinguished real from synthetic representations.

This indicates possible adversarial training imbalance.

The absolute diversity of generated samples nevertheless remained very low.

**TODO:** Insert verified training loss and diversity plots.

## 6. Baseline Methods

Two baseline methods were implemented.

### 6.1 Training Resample Baseline

Complete sequences are sampled randomly from the training dataset with replacement.

This method preserves authentic order-book structures because it reuses real sequences.

However, it does not generate genuinely novel market behaviour and should be interpreted as a distributional reference.

### 6.2 Repeated Training Mean Baseline

The mean training sequence is computed independently for each feature and time step.

The resulting template is then repeated to produce the required number of samples.

This baseline intentionally lacks sample diversity.

It is useful for identifying whether TimeGAN produces outputs that are only marginally more informative than a repeated average sequence.

## 7. Experimental Results

### 7.1 Distributional Comparison

All three methods were evaluated against the same held-out testing partition.

| Metric | TimeGAN V3.3 | Training Resample | Repeated Mean |
|---|---:|---:|---:|
| Mean MAE ↓ | 0.032308 | 0.030987 | 0.031123 |
| Std MAE ↓ | 0.065232 | 0.018556 | 0.073730 |
| Mean feature KL ↓ | 24.395889 | 0.356945 | 25.452350 |
| Market constraints ↑ | 97.292285% | 100% | 100% |
| Real–candidate SSIM | 0.399679 | 0.241607 | 0.397960 |
| Candidate–candidate SSIM | 0.999990 | 0.231964 | 1.000000 |

TimeGAN achieved marginally lower feature-wise KL divergence than the Repeated Mean baseline.

However, it performed substantially worse than the Training Resample reference.

The extremely high candidate-versus-candidate SSIM indicates that TimeGAN generated nearly identical depth structures across independent samples.

The reported feature-wise KL uses 50-bin histograms with numerical smoothing. The absolute magnitude depends on the estimation method.

### 7.2 Spread KL Divergence

The course specification requires KL divergence to be evaluated on bid–ask spreads and mid-price returns.

For each snapshot, the best bid–ask spread is:

\[
S_t=P^{ask}_{1,t}-P^{bid}_{1,t}
\]

The following results were obtained using 50-bin histogram KL estimation.

| Method | Spread KL |
|---|---:|
| TimeGAN V3.3 | 24.375349 |
| Training Resample | 0.347169 |
| Repeated Mean | 24.580134 |

The required target is KL ≤ 0.1.

TimeGAN did not achieve this target.

Its Spread KL was only slightly lower than the Repeated Mean baseline, indicating poor coverage of the real spread distribution.

The Training Resample reference also failed to reach the target, suggesting differences between training-period and testing-period spread distributions.

**Important limitation:** Mid-price Returns KL has not been measured for the generated sequences. The relative mid-price representation removes the price trajectory needed to generate independent mid-price returns.

Therefore, the project does not claim to have met the required return-distribution target.

### 7.3 Market Microstructure Validation

Generated sequences were inverse-transformed and checked against basic order-book constraints.

| Constraint | Real LOBSTER | TimeGAN |
|---|---:|---:|
| Positive best spread | 100.00% | 100.00% |
| Correct ask price ordering | 100.00% | 97.29% |
| Correct bid price ordering | 100.00% | 99.85% |
| Nonnegative quantities | 100.00% | 100.00% |
| All constraints satisfied | 100.00% | 97.29% |

The majority of generated snapshots satisfied the four implemented structural constraints.

However, approximately 2.71% of generated time steps contained an ask-price ordering violation.

These checks establish only basic market validity. Tick-size compliance, integer order quantities, and additional financial constraints were not evaluated.

### 7.4 Bid–Ask Spread Statistics

| Statistic | Real LOBSTER | TimeGAN |
|---|---:|---:|
| Mean spread | 0.097911 | 0.141463 |
| Spread standard deviation | 0.042384 | 0.015875 |

The generated mean spread was approximately 44.5% higher than the real test mean.

Its spread standard deviation was approximately 62.5% lower.

This indicates that generated spreads were systematically wider and less variable than those observed in the real test period.

### 7.5 Structural Similarity Evaluation

Depth heatmaps were constructed from 24 time steps and 20 bid/ask quantity features.

SSIM was computed using independent randomly selected samples.

| Comparison | Mean SSIM | Standard Deviation |
|---|---:|---:|
| Real vs Real | 0.251589 | 0.207440 |
| Real vs Generated | 0.399142 | 0.216332 |
| Generated vs Generated | 0.999992 | 0.000012 |

The Generated vs Generated SSIM is extremely close to 1.

This indicates that independently generated depth heatmaps are almost identical.

In contrast, real order-book sequences display substantially more variation.

The relatively high Real vs Generated score does not imply superior generative quality, since independent sequences do not have corresponding temporal targets.

A nearly constant generated pattern can resemble multiple real samples while failing to capture their diversity.

Therefore, the required SSIM > 0.6 structural-fidelity target has not been demonstrated by the current evaluation.

## 8. Failure Case Analysis

Five representative generated order-book sequences were selected for qualitative and quantitative error analysis.

Real and generated sequences with the same array index are independent samples rather than aligned forecasts.

### 8.1 Case Overview

| Failure Case | Index | Ask Violations | Real Mean Spread | Generated Mean Spread |
|---|---:|---:|---:|---:|
| Ask ladder violation | 0 | 1 | 0.17000 | 0.14174 |
| Wide spread | 460 | 1 | 0.15875 | 0.14205 |
| Flat depth | 589 | 0 | 0.12833 | 0.14040 |
| Flat spread | 1337 | 0 | 0.09917 | 0.14047 |
| Typical generated depth | 825 | 1 | 0.10958 | 0.14149 |

### 8.2 Case 1 — Ask Ladder Violation

**Generated sequence index: 0**

The generated sequence contains one time step in which the ask-price ladder is incorrectly ordered.

In a valid order book, ask prices should increase or remain nondecreasing across successive depth levels.

The violation indicates that an unconstrained continuous generator does not guarantee the preservation of financial market microstructure invariants.

**TODO:** Insert `ask_ladder_violation.png` and identify the precise price levels involved.

### 8.3 Case 2 — Wide Spread

**Generated sequence index: 460**

This sequence was selected because it has a relatively wide average spread within the generated sample collection.

Its generated mean spread is 0.14205, compared with 0.15875 in the independently selected real reference sequence.

Therefore, this particular pair does not demonstrate spread overestimation.

Instead, the overall spread overestimation conclusion is supported by the aggregate statistics.

This case illustrates the importance of distinguishing population-level distribution errors from differences between independent individual samples.

**TODO:** Insert `wide_spread.png`.

### 8.4 Case 3 — Flat Depth

**Generated sequence index: 589**

The sequence was selected for particularly low temporal variation in order-book depth quantities.

This is consistent with the near-perfect Generated vs Generated SSIM observed during quantitative evaluation.

The model appears to generate highly repetitive liquidity patterns rather than the diverse depth dynamics present in real order books.

This is a representative manifestation of mode collapse.

**TODO:** Insert `flat_depth.png` and inspect the measured depth variability.

### 8.5 Case 4 — Flat Spread Dynamics

**Generated sequence index: 1337**

The generated sequence exhibits unusually low spread variation.

Its mean spread is 0.14047, compared with 0.09917 in the independently selected reference sequence.

This example is consistent with the wider population-level finding that synthetic spreads are less variable than real spreads.

**TODO:** Insert `flat_spread.png` and report the sequence-specific spread standard deviation.

### 8.6 Case 5 — Typical Generated Depth

**Generated sequence index: 825**

This example was selected to represent a typical generated depth pattern rather than an extreme outlier.

Its generated mean spread is 0.14149.

It also contains one ask-price ordering violation.

This illustrates how synthetic snapshots can appear broadly plausible while remaining repetitive or violating price-level ordering.

**TODO:** Insert `typical_generated_depth.png`.

### 8.7 Summary of Failure Modes

The failure cases indicate several limitations:

1. Occasional violations of ask-price ordering.
2. Limited diversity in generated order-book depth.
3. Insufficient variation in bid–ask spreads.
4. Poor matching of real spread distributions.
5. Persistent mode collapse despite adjustments to training configuration.

These failures are important because structurally plausible outputs do not necessarily reproduce realistic financial market dynamics.

## 9. Computational Resource Analysis

### 9.1 Model Complexity and Training Cost

The final TimeGAN model contains **37,529 trainable parameters** across its five neural network components.

The final V3.3 experiment used CPU-based training, with two embedding epochs, two supervisor epochs, and ten joint adversarial training epochs.

| Measurement | Result |
|---|---:|
| Model parameters | 37,529 |
| Training device | CPU |
| Embedding pretraining epochs | 2 |
| Supervisor pretraining epochs | 2 |
| Joint training epochs | 10 |
| Total training time | 460.38 seconds |

### 9.2 Inference Performance

Inference performance was measured on CPU using a batch size of 64, with 10 warm-up batches followed by 100 timed batches.

| Measurement | Result |
|---|---:|
| Generated sequences | 6,400 |
| Time steps per sequence | 24 |
| Features per time step | 40 |
| Total inference time | 0.5928 seconds |
| Average latency per sequence | 0.0926 ms |
| Generation throughput | 10,795.45 sequences/second |

The reported latency represents average per-sequence throughput under batched inference, rather than standalone single-sequence inference latency.

These results indicate that the model can generate synthetic sequences efficiently on CPU. However, the quality evaluation demonstrates that computational efficiency alone does not guarantee realistic financial simulations.

### 9.3 GPU Resource Usage

The final V3.3 training and inference measurements were obtained on CPU. Therefore, peak GPU VRAM usage for the final model has not yet been measured.

A preliminary simplified GPU experiment was performed on an NVIDIA A100 GPU, but its resource measurements are not directly comparable to those of the final model.

**TODO:** Measure peak GPU VRAM for the full 40-feature TimeGAN model if GPU resources are available.

## 10. Reproducibility

### 10.1 Project Files

The implementation contains the following core files:

| File | Purpose |
|---|---|
| `modules.py` | TimeGAN neural network components |
| `dataset.py` | Original dataset loading and preprocessing |
| `dataset_v2.py` | Relative mid-price preprocessing |
| `modules_v3_2.py` | Identity-activation TimeGAN variant |
| `train.py` | Three-stage model training |
| `evaluate.py` | Initial distributional evaluation |
| `market_constraints.py` | Order-book structure validation |
| `ssim_evaluation.py` | Depth heatmap SSIM |
| `baseline_evaluation.py` | Baseline comparison |
| `failure_cases.py` | Representative failure case selection |
| `predict.py` | **TODO: Complete standalone inference and visualisation script** |
| `README.md` | Documentation and experimental report |

### 10.2 Dependencies

The implementation uses Python and the following libraries:

- PyTorch
- NumPy
- pandas
- Matplotlib
- scikit-image

**TODO:** Record exact tested versions and provide a reproducible `requirements.txt`.

### 10.3 Training

The final experiment can be reproduced using:

```bash
python train.py \
  --data-dir data \
  --dataset-version v2 \
  --model-version v3.2 \
  --epochs 2 \
  --supervisor-epochs 2 \
  --joint-epochs 10 \
  --joint-lr 0.0001 \
  --generator-updates 2 \
  --batch-size 64 \
  --device cpu \
  --output-dir checkpoints_v3_3_10ep
```

### 10.4 Evaluation

Run the initial distributional evaluation:

```bash
python evaluate.py \
  --data-dir data \
  --checkpoint checkpoints_v3_3_10ep/timegan_joint.pt \
  --device cpu \
  --output-dir outputs_v3_3_10ep
```

Run SSIM evaluation:

```bash
python ssim_evaluation.py \
  --data-dir data \
  --generated outputs_v3_3_10ep/generated_v2_v3_2.npy \
  --output-dir outputs_ssim_v3_3_10ep \
  --pairs 200 \
  --seed 42
```

Run baseline comparison:

```bash
python baseline_evaluation.py \
  --data-dir data \
  --generated outputs_v3_3_10ep/generated_v2_v3_2.npy \
  --output-dir outputs_baselines_v3_3 \
  --pairs 200 \
  --seed 42
```

Generate failure case visualisations:

```bash
python failure_cases.py \
  --data-dir data \
  --generated outputs_v3_3_10ep/generated_v2_v3_2.npy \
  --output-dir outputs_failure_cases_v3_3
```

The raw dataset, generated `.npy` files, and trained model checkpoints should not be committed to GitHub.

Only implementation scripts, necessary documentation, and appropriately sized report figures should be included in the repository.

## 11. Engineering Recommendation

Based on the current experiments, the final TimeGAN model should **not be deployed for realistic trading strategy backtesting**.

Although 97.29% of generated snapshots satisfy the four implemented market constraints, the model exhibits significant limitations.

The Spread KL of 24.375349 is substantially above the specified target of 0.1.

Generated depth heatmaps have an extremely high inter-sample SSIM, indicating severe diversity collapse.

The model also produces wider and less variable bid–ask spreads than the real test data.

Its additional architectural complexity therefore does not currently provide sufficient generative fidelity to justify operational deployment.

Future improvements should investigate:

- Explicit modelling of mid-price return dynamics.
- Diversity-preserving generative objectives.
- Improved adversarial training stability.
- Financial constraint-aware generation.
- Tick-size and quantity-validity checks.
- Evaluation across additional trading days.
- More extensive comparison with genuinely trained generative baselines.

The implementation remains useful as a research prototype for studying the strengths and limitations of adversarial financial time-series generation.

## 12. Artificial Intelligence Usage Disclosure

OpenAI ChatGPT was used to support learning, implementation, debugging, experimental planning, and technical documentation.

AI assistance included explanations of TimeGAN architecture, suggestions for PyTorch training code, data preprocessing, loss-function debugging, evaluation scripts, baseline construction, and failure-case analysis.

AI-generated suggestions were not assumed to be correct without verification.

The implementation was executed and tested on the Rangpur computing environment. Training logs, generated outputs, KL metrics, structural validity checks, baseline comparisons, and SSIM results were examined during development.

The final implementation and scientific claims remain the student's responsibility.

**TODO:** Add details of any additional independent code review, manual verification, or modifications actually performed.

## 13. References

**TODO:** Add verified references for:

1. The original TimeGAN publication.
2. The LOBSTER dataset and data documentation.
3. Any external code implementations or methodologies adapted in this project.
4. The SSIM evaluation method.
5. The COMP3710/COMP7710 Project 7 assessment specification.

Only sources genuinely consulted and used should be cited.

---

## 14. Conclusion

This project developed a complete prototype pipeline for synthetic limit order book generation using TimeGAN.

The pipeline includes dataset preprocessing, three-stage generative training, synthetic sequence generation, distributional evaluation, structural similarity analysis, market constraint validation, baseline comparisons, and representative failure-case analysis.

The experiments demonstrated that different preprocessing and optimisation strategies can influence training stability and generated sample variation.

However, the final model did not sufficiently reproduce the distributional diversity or spread dynamics of real order-book data.

In particular, mode collapse remained a major limitation, and the required market-distribution fidelity targets were not achieved.

The principal contribution of the work is a reproducible experimental framework for critically assessing synthetic financial order books rather than a deployment-ready market simulator.