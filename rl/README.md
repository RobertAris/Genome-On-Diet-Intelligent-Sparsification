# Genome-on-Diet RL Sandbox

Reinforcement learning environment for optimizing sparsification pattern generation in Genome-on-Diet read mapping.

## Overview

This RL sandbox trains a PPO agent to **generate** optimal sparsification patterns (length 1-6) for Illumina short read mapping. The policy network learns to create patterns that balance runtime and accuracy based on dataset characteristics.

## Environment

The `GenomeDietPatternEnv` is a Gymnasium environment that:

1. **Takes a MultiDiscrete action** (6 binary decisions: 0=skip, 1=keep for each position)
2. **Generates a pattern string** from the action (e.g., "101010", "111000", "110110")
3. **Runs Genome-on-Diet** with the generated pattern on a dataset
4. **Measures metrics**:
   - Runtime (seconds)
   - Mapping rate (% of reads mapped)
   - Average alignment score
   - Average edit distance
   - SNPs detected (count)
   - Indels detected (count)
   - Total variants detected (SNPs + indels)
5. **Returns a scalar reward** based on weighted combination of metrics

### Action Space

**MultiDiscrete[6]**: Policy network generates patterns of variable length (1-6)
- Each position is binary: `0` = skip this position, `1` = keep this position
- **Pattern encoding**: The rightmost '1' marks where the pattern ENDS
  - Pattern includes all positions from 0 up to (but not including) the rightmost '1'
- Example actions: 
  - `[1,1,0,0,0,0]` → pattern "1" (100% sparsification, length 1) - rightmost '1' at pos 1
  - `[1,0,1,0,0,0]` → pattern "10" (50% sparsification, length 2) - rightmost '1' at pos 2
  - `[1,0,0,1,0,0]` → pattern "100" (33% sparsification, length 3) - rightmost '1' at pos 3
  - `[1,0,1,1,0,0]` → pattern "101" (67% sparsification, length 3) - rightmost '1' at pos 3
- Total possible patterns: All combinations from length 1-6 (much larger search space)

The policy network learns to generate optimal patterns of variable length based on dataset features.

### Observation Space

4-dimensional continuous vector (dataset features only):
- `[0]`: Number of reads (normalized by 10K)
- `[1]`: Average read length in bp (normalized by 1K)
- `[2]`: Reference genome size in bp (normalized by 1B) - **varies per chromosome**
- `[3]`: GC content of reads (0-1)

**Note**: Performance metrics (runtime, mapping_rate, etc.) are outputs used for reward, not observations.

### Reward Function

The reward function combines multiple metrics with automatic normalization and hard constraints:

```
# Soft trade-offs (normalized):
reward = -2.5 × runtime                    # Speed (balanced with accuracy, not above it)
       + 1.5 × mapping_rate                # Mapping rate (moderate priority)
       + 0.0 × alignment_score             # Disabled (truth metrics are primary)
       + edit_distance_penalty             # Soft cap approach (see below)
       + truth_based_metrics               # F1, TP indels, FP penalty (PRIMARY accuracy)

# Edit Distance Soft Cap:
if edit_distance <= 3.0:
    edit_distance_penalty = -0.5 × edit_distance  # Small penalty (good enough for Illumina)
else:
    edit_distance_penalty = -0.5 × 3.0 - 4.0 × (edit_distance - 3.0)  # Quickly increasing penalty

# Hard constraints (applied after normalization):
if mapping_rate < 0.9:
    reward -= 3.0  # Large penalty (episode failed)
if edit_distance > 10.0:
    reward -= 3.0  # Large penalty (poor alignment quality)

# Soft penalty for low mapping rate:
if mapping_rate < 0.5:
    reward -= penalty_multiplier × (0.5 - mapping_rate)

# All weights normalized to ensure balanced contribution
reward = reward × normalization_factor
```

**Priority Order**: Truth metrics (F1, indels) > Runtime > Mapping rate > Edit distance (safety threshold)

**Edit Distance Strategy**: 
- **Soft cap at 3.0**: ED ≤ 3 gets small/no penalty (acceptable for Illumina)
- **Hard threshold at 10.0**: ED > 10 incurs large penalty (poor alignment quality)
- This gives the agent freedom to trade runtime vs ED as long as ED stays reasonable

**When `use_truth_metrics: true` (recommended):**
- `f1_score_weight = 4.0` (main global metric: balanced precision/recall - DOMINATES accuracy)
- `true_indels_tp_weight = 0.8` (strong emphasis: indels are most sensitive to sparsification quality)
- `false_positives_penalty = -1.0` (strong penalty: prevent "just call more" strategy)
- `true_snps_tp_weight = 0.0` (disabled: F1 already captures SNP performance)
- Raw counts disabled (`snp_count_weight = 0.0`, etc.) to prevent FP incentives
- `normalize_weights = true` (all weights normalized to ensure balanced contribution)
- `small_dataset_runtime_weight_multiplier = 0.3` (reduce runtime weight to 30% for small datasets)
- `small_dataset_accuracy_weight_multiplier = 1.5` (increase accuracy weights to 150% for small datasets)
- `small_dataset_threshold = 0.01` (threshold for "small dataset": normalized num_reads < 0.01 = <100 reads)

**Current weights (configurable in YAML, all normalized automatically):**
- `runtime_weight = -2.5` (balanced with accuracy, not above it)
- `mapping_rate_weight = 1.5` (moderate priority: prevent low-sensitivity patterns)
- `alignment_score_weight = 0.0` (disabled: truth metrics are primary accuracy measures)
- `edit_distance_weight = -0.5` (low continuous influence: soft cap approach)
- `edit_distance_soft_cap = 3.0` (ED ≤ 3: small/no penalty)
- `edit_distance_high_threshold = 3.0` (threshold for increasing penalty)
- `edit_distance_high_penalty_multiplier = 4.0` (strong extra penalty above soft cap)
- `edit_distance_hard_threshold = 10.0` (hard threshold: ED > 10 incurs large penalty)
- `mapping_rate_hard_threshold = 0.9` (hard threshold: mapping_rate < 0.9 incurs large penalty)

**Truth-Based Metrics (recommended when truth VCF available):**
When `use_truth_metrics = true`, the reward function uses truth-based variant metrics instead of raw counts:
- **Chromosome-filtered**: Truth variants are filtered to the current chromosome for accurate recall calculation
  - Recall denominator = truth variants for current chromosome only (not all 4M genome-wide)
- **Region-filtered**: Truth variants are further filtered to regions actually covered by reads
  - Extracts actual covered intervals from SAM file alignments (collects read intervals, merges overlaps)
  - Uses actual covered intervals, NOT a min/max bounding box (which would include huge gaps between sparse reads)
  - Only counts truth variants in positions that fall within any covered interval
  - This ensures meaningful recall values (e.g., 1,800 TP / 3,000 truth variants = 60%, not 66 / 315K = 0.02%)
  - Without interval filtering, recall is artificially near-zero because denominator includes entire chromosome
  - **chrX handling**: If no truth variants exist (e.g., HG002 v4.2.1 only has autosomes), truth metrics are set to 0
- **Scaled metrics**: F1, precision, and recall are scaled by 1000× to make them meaningful for learning
  - F1 typically 0.0006-0.0007 → scaled to 0.6-0.7 (meaningful gradients)
  - Without scaling, even weight 4.0 contributes only 0.0024 to reward (negligible)
  - With scaling, F1 weight 4.0 contributes 4.0 × 0.6 = 2.4 (meaningful!)
- **Sensitive variant calling**: bcftools uses permissive parameters to find more variants
  - `-Q 0`: Minimum base quality (very permissive, default is 13)
  - `-d 10000`: Max depth per BAM (avoids depth filtering)
  - `-C 50`: Adjust mapping quality (more sensitive)
- `true_snps_tp`: True positives for SNPs (compared against HG002 benchmark, chromosome-specific)
- `true_indels_tp`: True positives for indels (higher weight: indels more sensitive to sparsification)
- `false_positives`: False positive variants (penalty to discourage erroneous calls)
- `f1_score`: Balanced precision/recall metric (primary objective, scaled by 1000×)
- `precision`: Precision (TP / (TP + FP)), scaled by 1000×, weight 0.5
- `recall`: Recall (TP / (TP + FN)), scaled by 1000×, weight 0.5
- Raw variant counts are disabled (set to 0.0) to prevent rewarding false positives

**Dataset-Size-Aware Reward Adjustment (Indirect Approach):**
- For small datasets (few reads), the reward function adjusts weights:
  - **Runtime weight is reduced** (30% of normal) → speed matters less
  - **Accuracy weights are increased** (150% of normal) → accuracy matters more
- This indirectly encourages less sparsification by making accuracy more valuable than speed
- The model learns: **small datasets → prioritize accuracy → less sparsification naturally**
- Rationale: With fewer reads, accuracy is more critical than speed, so we can afford slower but more accurate patterns

## Installation

Install the required dependencies:

```bash
pip3 install -r rl/requirements.txt
```

Or install individually:

```bash
pip3 install stable-baselines3 gymnasium pyyaml numpy
```

## Documentation

- **[TRAINING_EXPLANATION.md](TRAINING_EXPLANATION.md)**: Comprehensive explanation of the training process, especially the Multi-Chromosome approach

## Usage

### 1. Prepare Multi-Chromosome Datasets (One-time setup)

```bash
cd /path/to/Genome-on-Diet

# Step 1a: Split full genome into chromosome files
python3 rl/scripts/split_genome_by_chromosome.py --chromosomes chr1 chr2 chr3 chr4 chr5 chr21 chr22 chrX

# Step 1b: Extract chromosome-specific reads with dataset size diversity
# Uses the full FASTQ file and extracts different numbers of reads per chromosome
# Large chromosomes get more reads, small chromosomes get fewer reads
# This creates diversity that helps the model learn: smaller datasets → less sparsification
python3 rl/scripts/extract_chromosome_reads.py \
    --sam-file <path-to-your-sam-file> \
    --input-fastq Data/D1_S1_L001_R1_001-017.fastq \
    --output-dir Data/chromosome_reads \
    --use-diversity-defaults
```

This creates:
- Individual chromosome FASTA files in `Data/chromosomes/`
- Chromosome-specific FASTQ files in `Data/chromosome_reads/` with varying read counts:
  - Large chromosomes (chr1, chr2): 1000 reads
  - Medium chromosomes (chr3, chr4, chr5): 600 reads
  - Small chromosomes (chr21, chr22): 150 reads
  - chrX: 300 reads

**Important:** Each chromosome's reads are mapped ONLY to that chromosome's reference region (e.g., chr1 reads → chr1.fasta). This ensures accurate mapping and realistic training.

### 2. Train with PPO (Multi-Chromosome)

```bash
cd /path/to/Genome-on-Diet
# Train with multi-chromosome datasets (recommended)
python3 rl/ppo_train.py --config rl/configs/env_multi_chromosome.yaml --timesteps 200

# Or use single dataset (legacy, deprecated)
python3 rl/ppo_train.py --timesteps 10000
```

**For multi-core systems:**
- Default: 6 threads per GDiet run (configurable in YAML)
- For faster training, use parallel environments:
  ```bash
  python3 rl/ppo_train.py --timesteps 10000 --n-envs 2 --use-subproc
  ```
  This runs 2 GDiet instances in parallel (each using 6 threads).
  - Total: ~12 threads across 2 processes
  - Leaves headroom for OS and other processes

### 3. Evaluate Trained Model

After training, evaluate your model:

```bash
# Basic evaluation (10 episodes)
python3 rl/evaluate_model.py

# Evaluate with more episodes
python3 rl/evaluate_model.py --n-episodes 20

# Compare against baseline pattern "10" (50% sparsification)
python3 rl/evaluate_model.py --n-episodes 20 --compare-baseline

# Use a different model
python3 rl/evaluate_model.py --model-path rl/models/ppo_genome_diet.zip --n-episodes 20
```

The evaluation script will show:
- Average reward, runtime, mapping rate, alignment score, edit distance
- Variant detection metrics (SNPs, indels, total variants)
- Pattern generation frequency (top patterns the agent generates)
- Episode-by-episode details
- Comparison with baseline pattern "10" (50% sparsification) if `--compare-baseline` is used

### 4. Configuration

Edit `rl/configs/env_default.yaml` to customize:
- Data paths (reference FASTA, query FASTQ)
- Pattern action space
- GDiet parameters
- Reward weights

## File Structure

```
rl/
├── envs/
│   ├── __init__.py
│   ├── genome_diet_pattern_env.py   # Main environment
│   └── dataset_manager.py           # Multi-chromosome dataset manager
├── configs/
│   ├── env_default.yaml             # Single-dataset config (deprecated)
│   └── env_multi_chromosome.yaml    # Multi-chromosome config (recommended)
├── scripts/
│   ├── extract_chromosome_reads.py  # Extract reads by chromosome
│   └── split_genome_by_chromosome.py # Split genome into chromosomes
├── ppo_train.py                     # PPO training script
├── evaluate_model.py                # Model evaluation script
├── README.md                         # This file
└── TRAINING_EXPLANATION.md           # Detailed training guide
```

## Outputs

During training, the following are generated:
- `rl/models/ppo_genome_diet.zip` - Trained PPO model (default)
- `rl/models/checkpoints/` - Periodic checkpoints (if checkpoint saving enabled)
- `rl/models/tensorboard/` - TensorBoard logs for monitoring training
- `rl/models/logs/` - Training logs (monitor.csv)
- `rl/outputs/` - Temporary SAM files (auto-cleaned after parsing)

## Truth-Based Metrics

When `use_truth_metrics: true` is enabled, the system uses benchmark-based evaluation:

1. **Variant Calling**: Calls variants from SAM files using bcftools with sensitive parameters
   - Uses permissive quality thresholds (`-Q 0`, `-d 10000`, `-C 50`) to find more variants
   - This addresses the issue where only 0.03% of truth variants were being found
2. **VCF Comparison**: Compares against HG002 truth VCF (cached at initialization for speed)
   - **Chromosome-filtered**: Only compares variants for the current chromosome
   - **Interval-filtered**: Only counts truth variants in positions within actual covered intervals (extracted from SAM)
     - Collects (start, end) intervals from each read's CIGAR
     - Merges overlapping intervals into non-overlapping chunks
     - Filters truth variants to positions within any interval (not a bounding box)
   - Ensures recall denominator is correct (e.g., 3K variants in covered intervals, not 315K for entire chromosome)
   - Without interval filtering, recall would be ~0.0002 (66 TP / 315K) instead of meaningful values like 0.60 (1,800 TP / 3K)
   - **chrX handling**: Returns default metrics (all zeros) when no truth data exists (HG002 v4.2.1 is autosomes only)
3. **Metrics**: Returns true positives (TP), false positives (FP), precision, recall, and F1 score
   - **Scaled by 1000×**: F1, precision, recall are multiplied by 1000 to make them meaningful
   - Without scaling, F1 ~0.0006 contributes only 0.0024 to reward (negligible)
   - With scaling, F1 ~0.6 contributes 2.4 to reward (meaningful learning signal)

**Performance**: Truth VCF is parsed once at environment initialization and cached, making subsequent comparisons fast.

**Note**: Variant calling adds ~10-30 seconds per episode. Consider disabling during training (`use_truth_metrics: false`) and enabling only for evaluation.

## Runtime Optimizations

Several optimizations have been implemented to reduce runtime:
- **Reference size caching**: FASTA sizes cached to avoid recalculation
- **Truth VCF caching**: 149MB truth VCF parsed once, reused for all comparisons
- **Efficient cleanup**: BAM files cleaned up after variant calling
- **Variable length patterns**: Patterns can be 1-6 in length (no need to always use length 6)
  - Pattern encoding: Rightmost '1' in action marks where pattern ends

## Notes

- The environment runs actual GDiet commands, so each step takes time (seconds to minutes depending on dataset size)
- SAM files are saved temporarily in `rl/outputs/` for metric extraction (auto-cleaned)
- The environment is single-step: each action terminates the episode
- Make sure GDiet is built and the executable path is correct in the config
- **Chromosome filtering**: Truth VCF variants are filtered to the current chromosome for accurate recall calculation. Without this, recall would be artificially near-zero (denominator would include all chromosomes 1-22). The recall denominator = truth variants for current chromosome only (e.g., ~250K per chromosome, not 4M genome-wide).
- **Interval filtering**: Truth variants are filtered to positions within actual covered intervals (extracted from SAM file read alignments). The system collects (start, end) intervals from each read's CIGAR, merges overlapping intervals, and only counts truth variants that fall within these intervals. This is critical because using a min/max bounding box would include huge gaps between sparse reads, making recall artificially low (~0.0002). With interval filtering, recall denominator = truth variants in actually covered positions only (e.g., ~3K variants), giving meaningful recall values (e.g., 60% instead of 0.02%).
- **chrX handling**: For chromosomes with no truth data (e.g., chrX in HG002 v4.2.1 which only includes autosomes 1-22), the system detects this and returns default truth metrics (all zeros), preventing division by zero and skipping truth-based reward contribution for those episodes.
- **Metric scaling**: Truth metrics (F1, precision, recall) are scaled by 1000× to make them meaningful for learning. This addresses the issue where F1 ~0.0006 was effectively constant noise, providing no learning signal.
- **Sensitive variant calling**: bcftools uses permissive parameters (`-Q 0`, `-d 10000`, `-C 50`) to find more variants. This addresses the issue where only 0.03% of truth variants were being found.
- **Pattern length**: Patterns are variable length (1-6). The rightmost '1' in the action marks where the pattern ends (pattern includes positions 0 to rightmost '1', exclusive)
- **Truth-based evaluation**: When `use_truth_metrics: true`, raw variant counts are disabled (set to 0.0) to prevent rewarding false positives. Only truth-based metrics (F1, TP indels, FP penalty) are used, and they are scaled by 1000× to provide meaningful gradients.

## Threading and Performance

### GDiet Threads (per environment)

The number of threads used by each GDiet run is controlled in `configs/env_default.yaml`:

```yaml
env:
  threads: 6  # Number of threads per GDiet run (adjust based on your CPU)
```

**Recommendations:**
- **8-core systems**: 6 threads (leaves 2 cores for OS/overhead)
- **10+ core systems**: 6-8 threads
- **4-6 core systems**: 2-4 threads

**Current Configuration:** All config files are set to use 6 threads by default.

### Parallel Environments

For faster training, you can run multiple environments in parallel:

```bash
# 2 parallel environments (each using 6 threads)
python rl/ppo_train.py --timesteps 10000 --n-envs 2 --use-subproc

# 4 parallel environments (may cause resource contention on smaller systems)
python rl/ppo_train.py --timesteps 10000 --n-envs 4 --use-subproc
```

**Note:** With `--use-subproc`, each environment runs in a separate process, allowing true parallelism. Without it, environments run sequentially even if you specify `--n-envs > 1`.

**Recommendations (assuming 6 threads per environment):**
- `--n-envs 1`: Single environment, 6 threads (baseline)
- `--n-envs 2` with `--use-subproc`: Recommended for 8-core systems (2× speedup, ~12 threads total)
- `--n-envs 4` with `--use-subproc`: May cause resource contention on smaller systems (24 threads total)

## TensorBoard Monitoring

TensorBoard is automatically enabled during training. To view metrics:

```bash
# Terminal 1: Start training
python3 rl/ppo_train.py --config rl/configs/env_multi_chromosome.yaml --timesteps 1000

# Terminal 2: Launch TensorBoard
tensorboard --logdir rl/models/tensorboard

# Browser: Open http://localhost:6006
```

**Key metrics to watch:**
- `rollout/ep_rew_mean`: Average episode reward (should increase)
- `train/policy_loss`: Policy loss (should decrease/stabilize)
- `train/value_loss`: Value function loss (should decrease)
- `train/entropy_loss`: Exploration (should decrease as model learns)

## Model Files

**`ppo_genome_diet.zip`**: Saved PPO model containing:
- Policy and value network weights
- Hyperparameters and training configuration
- Optimizer state

**Model compatibility**: Models trained with different configurations (action space, reward function) are **not compatible** - retrain required.

## Mapping Rate Notes

**With chromosome-specific reads** (current setup):
- Each episode uses reads from ONE chromosome mapped to that chromosome's reference
- Higher mapping rates expected (reads are pre-filtered to belong to the chromosome)
- More realistic training scenario

**For production**: Use full dataset (45K reads) with full genome for ~80-90% mapping rate.

## Project Status & Key Features

### Multi-Chromosome Training
- **8 chromosomes**: chr1, chr2, chr3, chr4, chr5, chr21, chr22, chrX
- **Diverse observations**: Different reference sizes (45MB-240MB) provide varied inputs
- **Context-aware learning**: Model learns to adapt patterns based on chromosome characteristics
- **Dataset size diversity**: Larger chromosomes have more reads, smaller chromosomes have fewer reads
  - This helps the model learn that smaller datasets should use less sparsification

### Reward Function Features
- **Truth metrics dominate**: F1 score (4.0) and indels (0.8) are PRIMARY accuracy measures
- **Edit distance soft cap**: ED ≤ 3 gets small/no penalty, ED > 3 gets quickly increasing penalty
- **Hard constraints**: Large penalties for failed episodes (mapping_rate < 0.9 or ED > 10)
- **Balanced runtime**: Runtime weight (-2.5) balanced with accuracy, not above it
- **Weight normalization**: All weights automatically normalized for balanced contribution
- **Dataset-size-aware adjustment**: Automatically adjusts weights for small datasets
- **False positive prevention**: Strong FP penalty (-1.0) prevents "just call more" strategy

### Current Architecture
- **Action Space**: MultiDiscrete[6] - generates patterns of variable length (1-6)
  - Encoding: Rightmost '1' marks pattern end (e.g., `[1,0,1,0,0,0]` → pattern "10")
- **Observation Space**: 4D vector `[num_reads, avg_read_length, reference_size, gc_content]`
- **Single-step episodes**: Each action = one complete GDiet run
- **PPO algorithm**: Proximal Policy Optimization with MLP policy network

### Technical Details
- **GDiet threads**: 6 threads per environment (configurable in YAML)
- **Parallel training**: Support for multiple parallel environments
- **SAM file parsing**: Extracts metrics including SNPs, indels, edit distance, alignment scores
- **TensorBoard logging**: Real-time monitoring of training progress

## Customization

### Changing Pattern Length

Edit `configs/env_multi_chromosome.yaml`:

```yaml
env:
  pattern_length: 6  # Maximum pattern length (patterns can be 1-6 in length)
```

### Adjusting Reward Weights

Modify the reward weights in `configs/env_default.yaml` or `configs/env_multi_chromosome.yaml` to emphasize different objectives:

```yaml
env:
  runtime_weight: -2.5        # Balanced with accuracy (not above it)
  mapping_rate_weight: 1.5   # Moderate priority: prevent low-sensitivity patterns
  alignment_score_weight: 0.0  # Disabled: truth metrics are primary accuracy measures
  edit_distance_weight: -0.5  # Low continuous influence: soft cap approach
  edit_distance_soft_cap: 3.0  # ED <= 3: small/no penalty (good enough for Illumina)
  edit_distance_high_threshold: 3.0  # Threshold for increasing penalty
  edit_distance_high_penalty_multiplier: 4.0  # Strong extra penalty above soft cap
  edit_distance_hard_threshold: 10.0  # Hard threshold: ED > 10 incurs large penalty
  edit_distance_hard_penalty: -3.0  # Large penalty for edit distance above hard threshold
  mapping_rate_hard_threshold: 0.9  # Hard threshold: mapping_rate < 0.9 incurs large penalty
  mapping_rate_hard_penalty: -3.0  # Large penalty for mapping rate below hard threshold
  
  # When use_truth_metrics: true, set these to 0.0 (disabled)
  # Raw counts reward quantity over quality and can incentivize false positives
  snp_count_weight: 0.0       # Disabled when truth metrics enabled
  indel_count_weight: 0.0     # Disabled when truth metrics enabled
  total_variants_weight: 0.0  # Disabled when truth metrics enabled
  
  # Truth-based weights (when use_truth_metrics: true) - PRIMARY accuracy measures
  true_indels_tp_weight: 0.8  # Strong emphasis: indels are most sensitive to sparsification quality
  f1_score_weight: 4.0       # Main global metric: balanced precision/recall (dominates accuracy)
  false_positives_penalty: -1.0  # Strong penalty: prevent "just call more" strategy
  small_dataset_runtime_weight_multiplier: 0.3  # Reduce runtime weight for small datasets (lower = speed matters less)
  small_dataset_accuracy_weight_multiplier: 1.5  # Increase accuracy weights for small datasets (higher = accuracy matters more)
  small_dataset_threshold: 0.01  # Threshold for "small dataset" (normalized num_reads)
```

**Dataset-Size-Aware Learning:**
- The reward function automatically encourages less sparsification for smaller datasets
- This is learned behavior, not hardcoded - the model discovers this through training
- Adjust `small_dataset_sparsification_penalty_weight` to control how strongly this effect is enforced

### Using Different Datasets

Update paths in `configs/env_default.yaml`:

```yaml
env:
  reference_fasta: Data/your_reference.fasta
  query_fastq: Data/your_reads.fastq
```


