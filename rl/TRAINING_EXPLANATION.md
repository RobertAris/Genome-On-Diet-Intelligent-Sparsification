# Training Process Explanation: Multi-Chromosome Approach

## Overview

The Genome-on-Diet RL training uses **Proximal Policy Optimization (PPO)** to learn optimal sparsification patterns. The key innovation is the **Multi-Chromosome Training Approach**, which provides diverse observations and enables the model to learn context-aware pattern selection.

## Training Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Initialize Environment                                     │
│    - Load config (YAML)                                       │
│    - Initialize ChromosomeDatasetManager                     │
│    - Create PPO model with MLP policy                         │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Training Loop (for each timestep)                        │
│                                                               │
│    a) Reset Environment                                      │
│       - Randomly select chromosome (chr1, chr2, etc.)        │
│       - Extract dataset features                             │
│       - Create observation vector                             │
│                                                               │
│    b) Agent Action                                           │
│       - Policy network generates pattern (variable length 1-6)│
│       - Encoding: Rightmost '1' marks pattern end            │
│       - Example: [1,1,0,0,0,0] → "1" (length 1)             │
│       - Example: [1,0,1,0,0,0] → "10" (length 2)            │
│       - Example: [1,0,1,0,1,1] → "10101" (length 5)         │
│                                                               │
│    c) Execute GDiet                                          │
│       - Run GDiet with generated pattern                     │
│       - Map reads to reference                                │
│       - Measure performance metrics                           │
│                                                               │
│    d) Compute Reward                                          │
│       - Combine runtime, mapping rate, alignment score, etc.  │
│       - If truth metrics enabled: use F1, TP indels, FP penalty│
│       - Raw variant counts disabled when truth metrics enabled │
│       - Truth VCF filtered to current chromosome for accuracy │
│       - All weights normalized automatically                   │
│                                                               │
│    e) Update Policy                                           │
│       - PPO collects n_steps (64) before updating            │
│       - Update policy and value networks                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Save Model                                                │
│    - Final model saved to rl/models/ppo_genome_diet.zip     │
│    - Checkpoints saved periodically (if enabled)             │
│    - TensorBoard logs for monitoring training                │
└─────────────────────────────────────────────────────────────┘
```

## Multi-Chromosome Approach: Deep Dive

### Why Multi-Chromosome Training?

**Problem with Single-Dataset Training:**
- Model sees the same observation every episode
- No diversity in dataset characteristics
- Model converges to a single pattern (overfitting)
- Poor generalization to new datasets

**Solution: Multi-Chromosome Training**
- Each episode uses a different chromosome dataset
- Diverse observations (different reference sizes, read characteristics)
- Model learns to adapt patterns based on dataset features
- Better generalization across different genome sizes

### How It Works

#### 1. Dataset Manager (`ChromosomeDatasetManager`)

The `ChromosomeDatasetManager` manages multiple chromosome datasets:

```python
# Initialization
manager = ChromosomeDatasetManager(
    chromosomes_dir="Data/chromosomes",      # Contains chr1.fasta, chr2.fasta, etc.
    query_fastq_dir="Data/chromosome_reads", # Contains chr1_reads.fastq, etc.
    chromosomes=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr21', 'chr22', 'chrX']
)

# Each episode: randomly select a chromosome
reference_fasta, query_fastq, chromosome = manager.get_random_dataset()
```

**Available Chromosomes:**
- **chr1**: 240.8 MB → `reference_size = 0.241` (largest)
- **chr2**: 234.3 MB → `reference_size = 0.234`
- **chr3**: 198.0 MB → `reference_size = 0.198`
- **chr4**: 190.2 MB → `reference_size = 0.190`
- **chr5**: 181.5 MB → `reference_size = 0.182`
- **chr21**: 45.2 MB → `reference_size = 0.045` (smallest)
- **chr22**: 50.8 MB → `reference_size = 0.051`
- **chrX**: 150.9 MB → `reference_size = 0.151`

#### 2. Episode Reset Process

Every time the environment resets (each episode):

```python
def reset(self):
    # Step 1: Randomly select a chromosome
    if self.config.dataset_manager is not None:
        self._current_reference_fasta, self._current_query_fastq, self._current_chromosome = (
            self.config.dataset_manager.get_random_dataset()
        )
    
    # Step 2: Extract dataset features from the selected chromosome
    self._dataset_features = self._extract_dataset_features(
        reference_fasta=self._current_reference_fasta,
        query_fastq=self._current_query_fastq,
    )
    
    # Step 3: Create observation vector
    observation = [
        num_reads / 10000.0,           # Normalized by 10K
        avg_read_length / 1000.0,      # Normalized by 1K bp
        reference_size / 1e9,          # Normalized by 1B bp (VARIES per chromosome!)
        gc_content                     # 0-1 range
    ]
    
    return observation, info
```

**Key Point:** The `reference_size` observation **varies** per chromosome, providing diverse inputs to the policy network.

#### 3. Context-Aware Pattern Selection

The model learns to generate different patterns based on the observation:

```
Observation: [num_reads, avg_read_length, reference_size, gc_content]
                    ↓
         Policy Network (MLP)
                    ↓
         Action: [1,0,1,0,1,1] → Pattern "10101" (rightmost '1' at pos 5, pattern is positions 0-5 exclusive = "10101")
```

**Example Learning:**
- **Small chromosome (chr21, reference_size=0.045)**: Model might learn pattern "1" or "11" (high sparsification, faster)
- **Large chromosome (chr1, reference_size=0.241)**: Model might learn pattern "10" or "10101" (balanced sparsification)

The model learns: *"For small references, use aggressive sparsification. For large references, use moderate sparsification."*

### Training Loop Details

#### PPO Configuration

```python
model = PPO(
    "MlpPolicy",           # Multi-layer perceptron policy
    vec_env,               # Vectorized environment (can be parallel)
    learning_rate=3e-4,    # Learning rate
    n_steps=64,            # Collect 64 timesteps before updating
    batch_size=32,         # Batch size for training
    n_epochs=4,            # Number of epochs per update
    gamma=0.99,           # Discount factor
    gae_lambda=0.95,      # GAE lambda
    clip_range=0.2,       # PPO clip range
    ent_coef=0.1,         # Entropy coefficient (exploration)
    vf_coef=0.5,          # Value function coefficient
)
```

#### Single-Step Episodes

**Important:** Each episode is a single step:
1. Agent receives observation (dataset features)
2. Agent generates action (pattern)
3. GDiet runs with pattern → metrics collected
4. Reward computed → episode terminates
5. Next episode starts with a new (random) chromosome

This is different from typical RL environments where episodes have multiple steps. Here, each action is a complete GDiet run.

#### Training Steps

```
Timestep 1:
  Episode 1: chr1 (reference_size=0.241) → Pattern "10101" → Reward: +2.5
  Episode 2: chr21 (reference_size=0.045) → Pattern "1" or "11" → Reward: +3.1
  Episode 3: chr2 (reference_size=0.234) → Pattern "10101" → Reward: +2.3
  ...
  (Collect 64 timesteps)
  
  → Update policy network (PPO update)
  
Timestep 65:
  Episode 65: chrX (reference_size=0.151) → Pattern "11011" → Reward: +2.8
  ...
  (Collect next 64 timesteps)
  
  → Update policy network (PPO update)
  
... (repeat until total_timesteps reached)
```

### Parallel Training

For faster training, you can use parallel environments:

```bash
python3 rl/ppo_train.py --n-envs 2 --use-subproc --timesteps 10000
```

**How it works:**
- Creates 2 separate processes (true parallelism)
- Each process runs GDiet independently
- Collects experiences in parallel
- Updates shared policy network

**Recommendations (assuming 6 threads per environment):**
- `--n-envs 1`: Single environment, 6 threads (baseline)
- `--n-envs 2 --use-subproc`: Recommended for 8-core systems (2× speedup, ~12 threads total)
- `--n-envs 4 --use-subproc`: May cause resource contention on smaller systems (24 threads total)

## Reward Function

The reward function combines multiple metrics with soft trade-offs and hard constraints:

```python
# Soft trade-offs (normalized):
reward = -2.5 × runtime                    # Speed (balanced with accuracy, not above it)
       + 1.5 × mapping_rate                  # Mapping rate (moderate priority)
       + 0.0 × alignment_score             # Disabled (truth metrics are primary)
       + edit_distance_penalty             # Soft cap approach (see below)
       + truth_based_metrics               # When use_truth_metrics=true:
                                          #   + 0.0 × true_snps_tp (disabled)
                                          #   + 0.8 × true_indels_tp (strong emphasis)
                                          #   + 4.0 × f1_score (main global metric)
                                          #   - 1.0 × false_positives (strong penalty)
                                          # Raw counts disabled (0.0) to prevent FP incentives

# Edit Distance Soft Cap:
if edit_distance <= 3.0:
    edit_distance_penalty = -0.5 × edit_distance  # Small penalty (good enough for Illumina)
else:
    edit_distance_penalty = -0.5 × 3.0 - 4.0 × (edit_distance - 3.0)  # Quickly increasing penalty

# Hard constraints (applied after normalization):
if mapping_rate < 0.9:
    reward -= 3.0  # Big penalty (episode failed)
if edit_distance > 10.0:
    reward -= 3.0  # Big penalty (alignment garbage)

# Soft penalty for low mapping rate:
if mapping_rate < 0.5:
    reward -= penalty_multiplier × (0.5 - mapping_rate)
```

**Key Points:**
- **Truth Metrics Dominate**: F1 score (4.0) and indels (0.8) are PRIMARY accuracy measures
- **Runtime Balanced**: Runtime weight (-2.5) balanced with accuracy, not above it
- **Edit Distance Soft Cap**: ED ≤ 3 gets small/no penalty, ED > 3 gets quickly increasing penalty
- **Hard Constraints**: Big penalties for failed episodes (mapping_rate < 0.9 or ED > 10)
- **False Positive Prevention**: Strong FP penalty (-1.0) prevents "just call more" strategy

## Benefits of Multi-Chromosome Training

### 1. **Diverse Observations**
- Different chromosome sizes → different `reference_size` values
- Model sees variety during training
- Prevents overfitting to a single dataset

### 2. **Context-Aware Learning**
- Model learns: *"Different datasets require different patterns"*
- Adapts pattern selection based on dataset characteristics
- Better generalization to unseen chromosomes

### 3. **Realistic Training**
- Each episode uses chromosome-specific reads
- Reads are pre-filtered to belong to the selected chromosome
- Higher mapping rates (more realistic than random reads)

### 4. **Better Exploration**
- Different chromosomes may favor different patterns
- Model explores pattern space more effectively
- Reduces premature convergence to a single pattern

## Training Output

### Files Generated

```
rl/models/
├── ppo_genome_diet.zip          # Final trained model (default save path)
├── checkpoints/                 # Periodic checkpoints (if checkpoint saving enabled)
│   └── ppo_genome_diet_5000_steps.zip
├── tensorboard/                 # TensorBoard logs for monitoring training
│   └── PPO_1/events.out.tfevents.*
├── logs/                        # Training logs (Monitor CSV)
│   └── monitor.csv
└── tensorboard/                 # TensorBoard logs
    └── PPO_7/
        └── events.out.tfevents.*
```

### Monitoring Training

**TensorBoard:**
```bash
tensorboard --logdir rl/models/tensorboard
# Open http://localhost:6006
```

**Key Metrics:**
- `rollout/ep_rew_mean`: Average episode reward (should increase)
- `rollout/ep_len_mean`: Average episode length (should be 1 for single-step)
- `train/policy_loss`: Policy loss (should decrease/stabilize)
- `train/value_loss`: Value function loss (should decrease)
- `train/entropy_loss`: Exploration (should decrease as model learns)

## Example Training Session

```bash
$ python3 rl/ppo_train.py --config rl/configs/env_multi_chromosome.yaml --timesteps 200

Dataset Manager initialized with 8 chromosomes:
  - chr1
  - chr2
  - chr21
  - chr22
  - chr3
  - chr4
  - chr5
  - chrX
Using 1 environment(s) (sequential)
Using cpu device
Starting PPO training for 200 timesteps...
Action space: MultiDiscrete([2 2 2 2 2 2])
Observation space: Box(-inf, inf, (4,), float32)
Pattern length: 6

Logging to rl/models/tensorboard/PPO_7
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | 2.23     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 1        |
|    time_elapsed    | 130      |
|    total_timesteps | 64       |
---------------------------------
...
Training complete! Model saved to: rl/models/ppo_genome_diet
```

## Comparison: Single vs Multi-Chromosome

| Aspect | Single Dataset | Multi-Chromosome |
|--------|---------------|------------------|
| **Observations** | Same every episode | Varies (different reference sizes) |
| **Pattern Diversity** | Converges to one pattern | Learns multiple patterns |
| **Generalization** | Poor (overfits) | Better (adapts to context) |
| **Training Data** | One chromosome | 8 chromosomes |
| **Learning** | "Always use pattern X" | "Use pattern X for small refs, Y for large refs" |

## Next Steps

1. **Train longer**: Use 10,000+ timesteps for better convergence
2. **Monitor diversity**: Track which patterns are selected for which chromosomes
3. **Evaluate**: Test on held-out chromosomes to verify generalization
4. **Truth-based metrics**: Enable `use_truth_metrics: true` for benchmark evaluation (slower but more accurate)
5. **Baseline comparison**: Use `--compare-baseline` to compare against pattern "10" (50% sparsification)

## Technical Details

### Chromosome-Specific Reads

Each chromosome has its own FASTQ file:
- `chr1_reads.fastq`: Reads that map to chr1
- `chr2_reads.fastq`: Reads that map to chr2
- etc.

**Generation:**
```bash
python3 rl/scripts/extract_chromosome_reads.py \
    --sam-file <path-to-your-sam-file> \
    --input-fastq Data/D1_S1_L001_R1_001-017_small.fastq \
    --output-dir Data/chromosome_reads
```

This ensures each episode uses reads that actually belong to the selected chromosome, leading to higher mapping rates and more realistic training.

### Observation Space Normalization

All observations are normalized for stable training:
- `num_reads`: Divided by 10,000
- `avg_read_length`: Divided by 1,000
- `reference_size`: Divided by 1,000,000,000 (1 billion)
- `gc_content`: Already 0-1 range

This ensures all features are in similar scales, preventing one feature from dominating the policy network.

