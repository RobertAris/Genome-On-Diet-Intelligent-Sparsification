# Data Files Purpose and Usage

This document explains the purpose of each data file and when to use them.

## Illumina Short Read Files

### `Data/D1_S1_L001_R1_001-017_small.fastq` (539KB, 1,000 reads)
**Purpose**: Fast RL training and testing
- **Use for**: 
  - Quick iteration during development
  - Testing reward functions
  - Prototyping new features
  - Fast training runs (~2-3 seconds per timestep)
- **Current usage**: Default in RL configs (`env_multi_chromosome.yaml`)

### `Data/D1_S1_L001_R1_001-017.fastq` (24MB, ~45,000 reads)
**Purpose**: Final training and realistic evaluation
- **Use for**:
  - **Final model training** (more realistic, better statistics)
  - **Production evaluation** (accurate metrics)
  - **Better generalization** (more diverse reads)
  - **Paper/experiment results** (realistic dataset size)
- **When to use**: After initial development, for final model training
- **Note**: Slower (~2-5 minutes per timestep) but more accurate

**To use for final training**, update config:
```yaml
query_fastq: Data/D1_S1_L001_R1_001-017.fastq  # Full dataset
```

## Reference Genome Files

### `Data/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta` (2.9GB)
**Purpose**: Full human genome reference
- **Use for**: 
  - Splitting into chromosome files
  - Full genome mapping (if needed)
  - Source for generating chromosome datasets
- **Current usage**: Input to `split_genome_by_chromosome.py`

### `Data/chromosomes/` directory
**Purpose**: Individual chromosome FASTA files for multi-chromosome training
- **Contains**: chr1.fasta, chr2.fasta, chr3.fasta, chr4.fasta, chr5.fasta, chr21.fasta, chr22.fasta, chrX.fasta
- **Use for**: 
  - Multi-chromosome RL training (provides diverse observations)
  - Testing on different chromosome sizes
- **Current usage**: Primary dataset for RL training (`env_multi_chromosome.yaml`)

### `Data/chromosome_reads/` directory (NEW)
**Purpose**: Chromosome-specific FASTQ files (one per chromosome)
- **Contains**: chr1_reads.fastq, chr2_reads.fastq, etc.
- **Generate using**: `python3 rl/scripts/extract_chromosome_reads.py`
- **Use for**: 
  - Each episode uses reads from ONE chromosome mapped to that chromosome's reference
  - Higher mapping rates (reads are pre-filtered to belong to the chromosome)
  - More realistic training scenario
- **Current usage**: Preferred method in `env_multi_chromosome.yaml` (via `query_fastq_dir`)

### `Data/GRCh38_chr1.fasta` (241MB)
**Purpose**: Single chromosome file for quick testing
- **Use for**:
  - Quick manual testing
  - Backward compatibility with deprecated `env_default.yaml`
  - Development/debugging
- **Note**: Can be regenerated from full genome if needed

## Training Workflow Recommendations

### Phase 0: Setup (One-time)
1. **Extract chromosome-specific reads**:
   ```bash
   # First, map full FASTQ to full genome to get chromosome assignments
   # Then extract reads by chromosome:
   python3 rl/scripts/extract_chromosome_reads.py \
       --sam-file <path-to-your-sam-file> \
       --input-fastq Data/D1_S1_L001_R1_001-017_small.fastq \
       --output-dir Data/chromosome_reads
   ```

### Phase 1: Development
- **Dataset**: Chromosome-specific FASTQ files in `Data/chromosome_reads/`
- **Reference**: `Data/chromosomes/` (multi-chromosome)
- **Purpose**: Fast iteration, testing reward functions, prototyping
- **Speed**: ~2-3 seconds per timestep
- **Mapping rate**: Higher (reads pre-filtered by chromosome)

### Phase 2: Final Training
- **Dataset**: `D1_S1_L001_R1_001-017.fastq` (45K reads)
- **Reference**: `Data/chromosomes/` (multi-chromosome) or full genome
- **Purpose**: Production-ready model, realistic evaluation
- **Speed**: ~2-5 minutes per timestep

### Phase 3: Evaluation
- **Dataset**: `D1_S1_L001_R1_001-017.fastq` (45K reads)
- **Reference**: Full genome or specific chromosomes
- **Purpose**: Final model validation, paper results

## File Size Summary

| File | Size | Reads | Use Case |
|------|------|-------|----------|
| `D1_S1_L001_R1_001-017_small.fastq` | 539KB | 1,000 | Development, fast training |
| `D1_S1_L001_R1_001-017.fastq` | 24MB | ~45,000 | Final training, evaluation |
| `GRCh38_chr1.fasta` | 241MB | - | Quick testing |
| `GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta` | 2.9GB | - | Full genome, chromosome source |
| `chromosomes/chr*.fasta` | 45MB-241MB each | - | Multi-chromosome training |

## Notes

- The **full Illumina FASTQ** is kept for future sophistication (final training)
- The **small version** is used for fast development
- Both can coexist - use small for development, full for final training
- Chromosome files are generated from the full genome as needed

