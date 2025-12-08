# Sparsification Pattern Parameters Guide

## Overview

Genome-on-Diet uses sparsification patterns to reduce the computational cost of sequence alignment by only considering a subset of positions in the sequences. This is controlled by two parameters:

- **`-Z STR`**: The diet pattern (a string of '1's and '0's)
- **`-W INT`**: The length of the diet pattern

## How Sparsification Patterns Work

The pattern is a repeating sequence that determines which positions to keep:
- **'1'** = Keep this position
- **'0'** = Skip this position

The pattern repeats across the entire sequence.

### Example 1: 50% Sparsification
```bash
-Z 10 -W 2
```
Pattern: `"10"` (length 2)
- Position 0: Keep (1)
- Position 1: Skip (0)
- Position 2: Keep (1)
- Position 3: Skip (0)
- ...

Result: Keeps 50% of bases (1 out of 2)

### Example 2: 67% Sparsification
```bash
-Z 110 -W 3
```
Pattern: `"110"` (length 3)
- Position 0: Keep (1)
- Position 1: Keep (1)
- Position 2: Skip (0)
- Position 3: Keep (1)
- Position 4: Keep (1)
- Position 5: Skip (0)
- ...

Result: Keeps 67% of bases (2 out of 3)

### Example 3: 67% Sparsification (Alternating Blocks)
```bash
-Z 111000111 -W 9
```
Pattern: `"111000111"` (length 9)
- Positions 0-2: Keep (111)
- Positions 3-5: Skip (000)
- Positions 6-8: Keep (111)
- ...

Result: Keeps 67% of bases (6 out of 9)

### Example 4: 75% Sparsification
```bash
-Z 1110 -W 4
```
Pattern: `"1110"` (length 4)
- Keeps 3 out of 4 bases (75%)

## Common Pattern Examples

| Pattern | W | Ones | Sparsification | Use Case |
|---------|---|------|----------------|----------|
| `10` | 2 | 1 | 50% | High speed, lower sensitivity |
| `110` | 3 | 2 | 67% | Balanced speed/sensitivity |
| `1110` | 4 | 3 | 75% | Higher sensitivity |
| `111000111` | 9 | 6 | 67% | Block-based pattern |
| `1` | 1 | 1 | 100% | No sparsification (full sequence) |

## Modifying Parameters in the Script

### Method 1: Environment Variables
```bash
export Z_PATTERN="110"
export W_LENGTH=3
export K_SIZE=21
export WINDOW_SIZE=11
./run_gdiet.sh
```

### Method 2: Inline Assignment
```bash
Z_PATTERN="1110" W_LENGTH=4 K_SIZE=23 ./run_gdiet.sh
```

### Method 3: Edit the Script
Modify the default values at the top of `run_gdiet.sh`:
```bash
DEFAULT_Z="110"      # Change pattern here
DEFAULT_W=3          # Change pattern length here
DEFAULT_K=21         # Change k-mer size
DEFAULT_WINDOW_SIZE=11  # Change window size
```

## Choosing the Right Pattern

### Factors to Consider:

1. **Speed vs. Sensitivity Trade-off**
   - Lower sparsification (fewer '1's) = Faster but may miss some alignments
   - Higher sparsification (more '1's) = Slower but more sensitive

2. **Read Type**
   - Short reads (Illumina): Often work well with 50-67% sparsification
   - Long reads (ONT/PacBio): May need higher sparsification (67-75%)

3. **Error Rate**
   - High error rates: Use higher sparsification to maintain sensitivity
   - Low error rates: Can use lower sparsification for speed

4. **Reference Complexity**
   - Repetitive genomes: May benefit from higher sparsification
   - Unique genomes: Can tolerate lower sparsification

## Testing Different Patterns

To compare different patterns, run the script multiple times with different parameters:

```bash
# Test 50% sparsification
Z_PATTERN="10" W_LENGTH=2 ./run_gdiet.sh

# Test 67% sparsification
Z_PATTERN="110" W_LENGTH=3 ./run_gdiet.sh

# Test 75% sparsification
Z_PATTERN="1110" W_LENGTH=4 ./run_gdiet.sh
```

Compare the mapping rates and alignment scores in the summary files to find the optimal pattern for your data.

## Pattern Validation

The pattern must:
- Contain only '1' and '0' characters
- Have length W matching the `-W` parameter
- Have at least one '1' (cannot skip all positions)

## Performance Impact

Sparsification affects:
- **Indexing time**: Reduced (fewer positions to index)
- **Mapping time**: Reduced (fewer positions to align)
- **Memory usage**: Reduced (smaller index)
- **Sensitivity**: May be reduced (fewer positions considered)

The exact impact depends on your specific data and hardware.

