"""Custom Gymnasium environment for Genome-on-Diet pattern selection RL."""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "gymnasium is required. Install via `pip install gymnasium`."
    ) from exc

import numpy as np

try:
    from .dataset_manager import ChromosomeDatasetManager
except ImportError:
    # Handle case where dataset_manager might not be available
    ChromosomeDatasetManager = None


@dataclass
class GenomeDietEnvConfig:
    """Configuration for the GenomeDietPatternEnv."""

    # Data paths - either single dataset or multi-dataset mode
    reference_fasta: Path = None  # Single dataset mode (deprecated, use dataset_manager)
    query_fastq: Path = None  # Single dataset mode (deprecated, use dataset_manager)
    
    # Multi-dataset mode (preferred)
    chromosomes_dir: Path = None  # Directory with chromosome FASTA files
    query_fastq_dir: Path = None  # Directory with chromosome-specific FASTQ files (chr1_reads.fastq, etc.)
    dataset_manager: ChromosomeDatasetManager = None  # Pre-initialized dataset manager
    
    gdiet_executable: Path = Path("GDiet-ShortReads/GDiet")
    
    # Pattern generation - maximum length of pattern (patterns can be 1 to pattern_length)
    pattern_length: int = 6  # Maximum pattern length (patterns can be variable length <= 6)
    
    # GDiet parameters
    k_size: int = 21
    window_size: int = 11
    threads: int = 1
    max_seeds: int = 2
    secondary: int = 1
    bandwidth: str = "0.05,150,200"
    min_cnt: str = "0.95,0.3"
    min_score: int = 100
    af_max_loc: int = 2
    
    # Reward weights (will be normalized)
    # Priority: Truth metrics (F1, indels) > Runtime > Mapping rate > Edit distance (safety threshold)
    runtime_weight: float = -2.5  # Important but balanced: minimize runtime (on par with accuracy, not above it)
    mapping_rate_weight: float = 1.5  # Moderate priority: prevent low-sensitivity patterns
    alignment_score_weight: float = 0.0  # Disabled: truth metrics are primary accuracy measures
    edit_distance_weight: float = -0.5  # Low continuous influence: soft cap approach
    edit_distance_soft_cap: float = 3.0  # ED <= 3: small/no penalty (good enough for Illumina)
    edit_distance_high_threshold: float = 3.0  # Threshold for "high" edit distance (moved from 20 to 3)
    edit_distance_high_penalty_multiplier: float = 4.0  # Strong extra penalty above soft cap
    edit_distance_hard_threshold: float = 10.0  # Hard threshold: ED > 10 gets big penalty (alignment garbage)
    # Raw variant count weights (ONLY used when use_truth_metrics=False)
    # WARNING: These reward quantity over quality - can incentivize false positives!
    # When truth metrics are available (use_truth_metrics=True), these should be 0.0
    # Default to 0.0 to prevent accidental use when truth metrics are enabled
    snp_count_weight: float = 0.0  # Disabled by default: use truth metrics instead
    indel_count_weight: float = 0.0  # Disabled by default: use truth metrics instead
    total_variants_weight: float = 0.0  # Disabled by default: use truth metrics instead
    min_mapping_rate_threshold: float = 0.5  # Soft penalty if mapping rate below this threshold
    mapping_rate_hard_threshold: float = 0.9  # Hard threshold: mapping_rate < 0.9 gets big penalty (episode failed)
    mapping_rate_hard_penalty: float = -3.0  # Big penalty for mapping rate below hard threshold
    edit_distance_hard_penalty: float = -3.0  # Big penalty for edit distance above hard threshold
    
    # Truth-based metric weights (used when use_truth_metrics=True, will be normalized)
    # When truth metrics are available, these should be primary accuracy measures
    # Raw count weights (snp_count_weight, etc.) should be 0.0 when truth metrics are used
    true_snps_tp_weight: float = 0.0  # Optional: F1 score already captures SNP performance
    true_indels_tp_weight: float = 0.8  # Strong emphasis: indels are first casualty of bad sparsification
    f1_score_weight: float = 4.0  # Main global metric: balanced precision/recall (dominates accuracy)
    false_positives_penalty: float = -1.0  # Strong penalty: prevent "just call more" strategy
    normalize_weights: bool = True  # Whether to normalize all weights to sum to a target magnitude
    
    # Dataset-size-aware reward adjustment (indirect approach)
    # For small datasets: reduce runtime weight (speed matters less) and increase accuracy weights
    # This indirectly encourages less sparsification by making accuracy more important than speed
    small_dataset_runtime_weight_multiplier: float = 0.3  # Reduce runtime weight for small datasets (0.3 = 30% of normal)
    small_dataset_accuracy_weight_multiplier: float = 1.5  # Increase accuracy weights for small datasets (1.5 = 150% of normal)
    small_dataset_threshold: float = 0.01  # Threshold for "small dataset" (normalized num_reads < 0.01 = <100 reads)
    
    # Observation space
    # Only dataset features that the agent can see BEFORE making a decision
    # Performance metrics (runtime, mapping_rate, etc.) are outputs used for reward, not observations
    observation_dim: int = 4  # num_reads, avg_read_length, reference_size, gc_content
    
    # Output directory for temporary SAM files
    output_dir: Path = Path("rl/outputs")
    
    # Truth-based evaluation (optional)
    truth_vcf: Path = None  # Path to truth VCF file (e.g., HG002 benchmark)
    confident_regions_bed: Path = None  # Path to confident regions BED file (optional, for hap.py)
    use_truth_metrics: bool = False  # Enable truth-based metrics (requires truth_vcf)
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate pattern length
        if self.pattern_length < 2:
            raise ValueError(f"pattern_length must be at least 2, got {self.pattern_length}")
        
        # Convert paths to absolute
        self.gdiet_executable = Path(self.gdiet_executable).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle dataset manager initialization
        if self.dataset_manager is None and self.chromosomes_dir is not None:
            # Initialize dataset manager if chromosomes_dir is provided
            if self.query_fastq_dir is None and self.query_fastq is None:
                raise ValueError("Either query_fastq_dir or query_fastq must be provided when using chromosomes_dir")
            self.dataset_manager = ChromosomeDatasetManager(
                chromosomes_dir=self.chromosomes_dir,
                query_fastq=self.query_fastq,
                query_fastq_dir=self.query_fastq_dir,
            )
        
        # Legacy single-dataset mode
        if self.reference_fasta is not None and self.query_fastq is not None:
            self.reference_fasta = Path(self.reference_fasta).expanduser().resolve()
            self.query_fastq = Path(self.query_fastq).expanduser().resolve()

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], project_root: Path | None = None) -> "GenomeDietEnvConfig":
        """
        Create config from dictionary.
        
        Args:
            cfg: Configuration dictionary
            project_root: Root directory of the project (for resolving relative paths).
                         If None, tries to infer from common locations.
        """
        cfg = cfg.copy()
        
        # Try to find project root if not provided
        if project_root is None:
            # Look for common project root indicators
            current = Path.cwd()
            if (current / "GDiet-ShortReads").exists():
                project_root = current
            elif (current.parent / "GDiet-ShortReads").exists():
                project_root = current.parent
            else:
                project_root = current  # Fallback to current directory
        
        # Handle path conversions - resolve relative to project root
        for key in ["reference_fasta", "query_fastq", "chromosomes_dir", "query_fastq_dir", "gdiet_executable", "output_dir", "truth_vcf", "confident_regions_bed"]:
            if key in cfg and cfg[key] is not None:
                path = Path(cfg[key])
                if not path.is_absolute():
                    path = (project_root / path).resolve()
                else:
                    path = path.expanduser().resolve()
                cfg[key] = path
        
        return cls(**cfg)


class GenomeDietPatternEnv(gym.Env):
    """
    Gymnasium environment for learning optimal sparsification patterns.
    
    Action: MultiDiscrete[6] -> generates a pattern of variable length 1-6 (0=skip, 1=keep for each position)
                 Rightmost '1' marks where pattern ends (pattern includes positions 0 to rightmost '1', exclusive)
    Observation: [num_reads, avg_read_length, reference_size, gc_content]
                 - Only dataset features that agent can see BEFORE making a decision
                 - Performance metrics (runtime, mapping_rate, etc.) are outputs used for reward
    Reward: weighted combination of runtime and accuracy metrics (computed from GDiet results)
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, config: GenomeDietEnvConfig) -> None:
        super().__init__()
        self.config = config
        
        # Initialize state tracking variables FIRST
        self._current_observation = np.zeros(self.config.observation_dim, dtype=np.float32)
        self._episode_count = 0
        self._current_reference_fasta = None
        self._current_query_fastq = None
        self._current_chromosome = None
        self._dataset_features = None
        
        # Cache truth VCF variants if truth metrics are enabled (parse once, reuse)
        self._truth_vcf_cache = None
        # Cache reference sizes to avoid recalculating
        self._reference_size_cache = {}
        if self.config.use_truth_metrics and self.config.truth_vcf and self.config.truth_vcf.exists():
            print(f"Loading truth VCF cache from {self.config.truth_vcf}...")
            self._truth_vcf_cache = self._parse_vcf_variants(self.config.truth_vcf)
            print(f"✓ Cached {len(self._truth_vcf_cache)} truth variants")
        
        # Validate configuration and set initial dataset
        if self.config.dataset_manager is None:
            # Single dataset mode
            if self.config.reference_fasta is None or self.config.query_fastq is None:
                raise ValueError(
                    "Either dataset_manager or (reference_fasta, query_fastq) must be provided"
                )
            self._current_reference_fasta = self.config.reference_fasta
            self._current_query_fastq = self.config.query_fastq
            self._current_chromosome = "single_dataset"
        
        # Validate paths
        if not self.config.gdiet_executable.exists():
            raise FileNotFoundError(
                f"GDiet executable not found: {self.config.gdiet_executable}"
            )
        
        # Validate initial dataset (for single-dataset mode)
        if self._current_reference_fasta and not self._current_reference_fasta.exists():
            raise FileNotFoundError(
                f"Reference FASTA not found: {self._current_reference_fasta}"
            )
        if self._current_query_fastq and not self._current_query_fastq.exists():
            raise FileNotFoundError(
                f"Query FASTQ not found: {self._current_query_fastq}"
            )
        
        # Action space: MultiDiscrete for pattern generation (maximum length 6, variable actual length)
        # Each position is binary: 0 = skip, 1 = keep
        # Pattern encoding: The rightmost '1' marks where the pattern ENDS.
        # Pattern includes all positions from 0 up to (but not including) the rightmost '1'.
        # Examples: [1,1,0,0,0,0] → "1", [1,0,1,0,0,0] → "10", [1,0,0,1,0,0] → "100"
        self.action_space = spaces.MultiDiscrete([2] * self.config.pattern_length)
        
        # Observation space: normalized metrics
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.observation_dim,),
            dtype=np.float32,
        )

    def _action_to_pattern_string(self, action: np.ndarray) -> str:
        """Convert action array to pattern string. Supports variable length <= 6.
        
        IMPORTANT: Preserves trailing zeros - "10" is different from "1", "100" is different from "1".
        
        Pattern encoding: The rightmost '1' in the action array marks the END of the pattern.
        The pattern includes all positions from 0 up to (but not including) that rightmost '1'.
        
        Examples:
        - Pattern "1": [1, 1, 0, 0, 0, 0] → rightmost '1' at position 1 → pattern is positions 0 to 1 (exclusive) → "1"
        - Pattern "10": [1, 0, 1, 0, 0, 0] → rightmost '1' at position 2 → pattern is positions 0 to 2 (exclusive) → "10"
        - Pattern "100": [1, 0, 0, 1, 0, 0] → rightmost '1' at position 3 → pattern is positions 0 to 3 (exclusive) → "100"
        - Pattern "101": [1, 0, 1, 1, 0, 0] → rightmost '1' at position 3 → pattern is positions 0 to 3 (exclusive) → "101"
        - Pattern "11": [1, 1, 1, 0, 0, 0] → rightmost '1' at position 2 → pattern is positions 0 to 2 (exclusive) → "11"
        
        Special case: If the rightmost '1' is at position 0, the pattern is just "1" (no marker needed).
        """
        # Action is array of 0s and 1s, convert to string
        pattern_list = [str(int(a)) for a in action]
        
        # Find the rightmost '1' - this marks where the pattern ENDS
        last_one_idx = -1
        for i in range(len(pattern_list) - 1, -1, -1):
            if pattern_list[i] == '1':
                last_one_idx = i
                break
        
        # If no '1' found, use minimum pattern "1" (shouldn't happen with valid actions)
        if last_one_idx == -1:
            return "1"
        
        # If rightmost '1' is at position 0, pattern is just "1" (no trailing marker)
        if last_one_idx == 0:
            return "1"
        
        # Pattern includes all positions from 0 to rightmost '1' (exclusive)
        # The rightmost '1' is a marker indicating where the pattern ends
        pattern = ''.join(pattern_list[:last_one_idx])
        
        # Ensure at least one position is kept (should always have at least the first position)
        if len(pattern) == 0 or '1' not in pattern:
            return "1"
        
        return pattern
    
    def _extract_dataset_features(self, reference_fasta: Path = None, query_fastq: Path = None) -> Dict[str, float]:
        """
        Extract dataset characteristics from FASTQ and FASTA files.
        
        Returns:
            Dictionary with normalized dataset features:
            - num_reads: Number of reads (normalized by 10000)
            - avg_read_length: Average read length in base pairs
            - reference_size: Reference genome size in base pairs (normalized by 1e9)
            - gc_content: GC content of reads (0-1)
        """
        features = {}
        
        # Use provided paths or current dataset paths
        query_fastq = query_fastq or self._current_query_fastq or self.config.query_fastq
        reference_fasta = reference_fasta or self._current_reference_fasta or self.config.reference_fasta
        
        # Extract FASTQ features
        try:
            with open(query_fastq, "r") as f:
                lines = f.readlines()
            
            # Count reads: FASTQ format is 4 lines per read
            # Line 0: @header
            # Line 1: sequence
            # Line 2: + (optional header)
            # Line 3: quality scores
            # Count sequence lines (every 4th line starting from index 1)
            sequence_lines = [line.strip() for i, line in enumerate(lines) if i % 4 == 1]
            num_reads = len(sequence_lines)
            
            # Extract read lengths and compute average
            read_lengths = []
            gc_counts = {"G": 0, "C": 0, "A": 0, "T": 0, "N": 0, "total": 0}
            
            for seq in sequence_lines:
                seq = seq.upper()
                read_lengths.append(len(seq))
                # Count GC content
                for base in seq:
                    if base in gc_counts:
                        gc_counts[base] += 1
                        gc_counts["total"] += 1
            
            avg_read_length = np.mean(read_lengths) if read_lengths else 0.0
            
            # Compute GC content
            if gc_counts["total"] > 0:
                gc_content = (gc_counts["G"] + gc_counts["C"]) / gc_counts["total"]
            else:
                gc_content = 0.0
            
            # Normalize features
            features["num_reads"] = num_reads / 10000.0  # Normalize by 10K reads
            features["avg_read_length"] = avg_read_length / 1000.0  # Normalize by 1K bp
            features["gc_content"] = gc_content
            
        except Exception as e:
            print(f"Warning: Failed to extract FASTQ features: {e}")
            features["num_reads"] = 0.0
            features["avg_read_length"] = 0.0
            features["gc_content"] = 0.0
        
        # Extract FASTA features (reference size) - use cache if available
        try:
            ref_path_str = str(reference_fasta)
            if ref_path_str in self._reference_size_cache:
                features["reference_size"] = self._reference_size_cache[ref_path_str]
            else:
                total_bases = 0
                with open(reference_fasta, "r") as f:
                    for line in f:
                        if not line.startswith(">"):
                            total_bases += len(line.strip())
                
                features["reference_size"] = total_bases / 1e9  # Normalize by 1 billion bp
                self._reference_size_cache[ref_path_str] = features["reference_size"]
            
        except Exception as e:
            print(f"Warning: Failed to extract FASTA features: {e}")
            features["reference_size"] = 0.0
        
        return features

    def _run_gdiet(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Run Genome-on-Diet with the generated pattern and return runtime and metrics.
        
        Args:
            action: Array of binary decisions (0=skip, 1=keep) for each position
        
        Returns:
            (runtime_seconds, metrics_dict)
        """
        pattern_str = self._action_to_pattern_string(action)
        pattern_length = len(pattern_str)
        
        # Create temporary output file
        pattern_str = self._action_to_pattern_string(action)
        output_sam = self.config.output_dir / f"gdiet_pattern_{pattern_str}_ep{self._episode_count}.sam"
        
        # Use current dataset paths
        reference_fasta = self._current_reference_fasta or self.config.reference_fasta
        query_fastq = self._current_query_fastq or self.config.query_fastq
        
        # Build GDiet command
        cmd = [
            str(self.config.gdiet_executable),
            "-t", str(self.config.threads),
            "-x", "sr",  # short reads
            "-Z", pattern_str,
            "-W", str(pattern_length),
            "-k", str(self.config.k_size),
            "-w", str(self.config.window_size),
            "-i", str(self.config.max_seeds),
            "-N", str(self.config.secondary),
            "-r", self.config.bandwidth,
            "-n", self.config.min_cnt,
            "-s", str(self.config.min_score),
            "--AF_max_loc", str(self.config.af_max_loc),
            "--secondary=yes",
            "--MD",  # Enable MD tag for variant detection
            "-a",
            "-o", str(output_sam),
            str(reference_fasta),
            str(query_fastq),
        ]
        
        # Run GDiet and measure runtime
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600,  # 1 hour timeout
            )
            runtime = time.time() - start_time
        except subprocess.TimeoutExpired:
            runtime = 3600.0  # Max timeout
            metrics = self._default_metrics()
            return runtime, metrics
        except subprocess.CalledProcessError as e:
            # If GDiet fails, return default metrics with penalty
            runtime = time.time() - start_time
            print(f"Warning: GDiet failed with pattern {pattern_str}: {e.stderr}")
            metrics = self._default_metrics()
            return runtime, metrics
        
        # Parse SAM file to extract metrics
        metrics = self._parse_sam_metrics(output_sam)
        
        # If truth-based metrics are enabled, call variants and compare
        # Only run variant calling periodically to save time (every 5th episode or on evaluation)
        if self.config.use_truth_metrics and self.config.truth_vcf:
            # Skip variant calling during training for speed (only use during evaluation)
            # Set use_truth_metrics=False during training if too slow
            query_vcf = self._call_variants_from_sam(output_sam, reference_fasta)
            if query_vcf:
                # Get current chromosome for filtering truth variants
                current_chromosome = self._current_chromosome or self._extract_chromosome_from_fasta(reference_fasta)
                
                # CRITICAL: Extract coverage regions from SAM to filter truth variants
                # This ensures recall denominator only includes variants in regions actually sequenced
                # Without this, recall is artificially near-zero (~0.0002) because denominator includes
                # all truth variants on the chromosome, even in regions never sequenced
                coverage_regions = self._extract_coverage_regions_from_sam(output_sam, current_chromosome)
                
                truth_metrics = self._compare_vcf_with_truth(
                    query_vcf,
                    self.config.truth_vcf,
                    reference_fasta,
                    self.config.confident_regions_bed,
                    current_chromosome=current_chromosome,
                    coverage_regions=coverage_regions,
                )
                metrics.update(truth_metrics)
                
                # Clean up VCF file and intermediate BAM files
                try:
                    query_vcf.unlink()
                    # Clean up BAM files created during variant calling
                    bam_pattern = output_sam.stem
                    for bam_file in self.config.output_dir.glob(f"{bam_pattern}*.bam*"):
                        try:
                            bam_file.unlink()
                        except Exception:
                            pass
                except Exception:
                    pass
        
        # Clean up SAM file after parsing (keep outputs/ directory clean)
        try:
            output_sam.unlink()
        except Exception:
            pass  # Ignore cleanup errors
        
        return runtime, metrics

    def _parse_cigar_variants(self, cigar_str: str) -> Dict[str, int]:
        """
        Parse CIGAR string to count indels and SNPs.
        
        CIGAR operations:
        - M: match/mismatch (can contain SNPs)
        - I: insertion (indel)
        - D: deletion (indel)
        - N: skipped region
        - S: soft clip
        - H: hard clip
        - P: padding
        - =: match
        - X: mismatch (SNP)
        
        Returns:
            Dictionary with indel_count and snp_count
        """
        indels = 0
        snps = 0
        
        # Parse CIGAR string (e.g., "100M2I50M3D20M")
        import re
        cigar_ops = re.findall(r'(\d+)([MIDNSHPX=])', cigar_str)
        
        for length_str, op in cigar_ops:
            length = int(length_str)
            if op == 'I' or op == 'D':
                # Count each indel event (not total bases)
                indels += 1
            elif op == 'X':
                # Explicit mismatch (SNP)
                snps += length
            elif op == 'M':
                # M can be match or mismatch - we'll estimate SNPs from edit distance
                # For now, we count M as potential SNPs (will refine with MD tag)
                pass
        
        return {"indel_count": indels, "snp_count": snps}
    
    def _count_indel_bases(self, cigar_str: str) -> int:
        """
        Count total bases affected by indels (not just events).
        
        Returns:
            Total number of bases in insertions and deletions
        """
        import re
        cigar_ops = re.findall(r'(\d+)([MIDNSHPX=])', cigar_str)
        indel_bases = 0
        
        for length_str, op in cigar_ops:
            if op == 'I' or op == 'D':
                indel_bases += int(length_str)
        
        return indel_bases
    
    def _parse_md_tag_for_snps(self, md_tag: str, cigar_str: str) -> int:
        """
        Parse MD tag to count SNPs more accurately.
        
        MD tag format: "MD:Z:100^AT10G5" means:
        - 100 matches
        - deletion of AT
        - 10 matches
        - SNP: G (reference) -> (query has different base)
        - 5 matches
        
        Returns:
            Number of SNPs
        """
        if not md_tag or not md_tag.startswith("MD:Z:"):
            return 0
        
        md_str = md_tag.split(":")[2]
        snp_count = 0
        
        # Count single base mismatches (not indels)
        # MD format: numbers = matches, letters = mismatches, ^ = deletion
        import re
        # Find all single letter patterns (SNPs) not preceded by ^
        # Simple approach: count letters that aren't part of ^deletion
        parts = re.split(r'\^[ACGTN]+', md_str)  # Remove deletions
        for part in parts:
            # Count letters (mismatches/SNPs) in remaining parts
            snp_count += len(re.findall(r'[ACGTN]', part))
        
        return snp_count
    
    def _call_variants_from_sam(self, sam_file: Path, reference_fasta: Path) -> Path:
        """
        Call variants from SAM file using bcftools mpileup and call.
        
        Returns:
            Path to generated VCF file
        """
        output_vcf = self.config.output_dir / f"variants_{sam_file.stem}.vcf"
        
        try:
            # Step 1: Convert SAM to BAM and sort
            bam_file = self.config.output_dir / f"{sam_file.stem}.bam"
            sorted_bam = self.config.output_dir / f"{sam_file.stem}.sorted.bam"
            
            # Check if samtools is available
            try:
                subprocess.run(["samtools", "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Warning: samtools not found. Skipping variant calling.")
                return None
            
            # Convert SAM to BAM
            with open(sam_file, "r") as sam_in, open(bam_file, "wb") as bam_out:
                result = subprocess.run(
                    ["samtools", "view", "-bS", "-"],
                    stdin=sam_in,
                    stdout=bam_out,
                    stderr=subprocess.PIPE,
                    check=True,
                )
            
            # Sort BAM
            subprocess.run(
                ["samtools", "sort", "-o", str(sorted_bam), str(bam_file)],
                check=True,
            )
            
            # Check if BAM has any mapped reads
            count_result = subprocess.run(
                ["samtools", "view", "-c", "-F", "4", str(sorted_bam)],
                capture_output=True,
                text=True,
                check=True,
            )
            mapped_count = int(count_result.stdout.strip())
            if mapped_count == 0:
                print("Warning: No mapped reads in SAM file. Skipping variant calling.")
                # Cleanup
                try:
                    bam_file.unlink()
                    sorted_bam.unlink()
                except Exception:
                    pass
                return None
            
            # Index BAM
            subprocess.run(
                ["samtools", "index", str(sorted_bam)],
                check=True,
            )
            
            # Step 2: Call variants using bcftools mpileup and call
            try:
                subprocess.run(["bcftools", "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Warning: bcftools not found. Skipping variant calling.")
                return None
            
            # Create VCF with mpileup and call
            # CRITICAL: Use more sensitive parameters to detect variants
            # Default bcftools is too conservative and misses most variants
            # -Q 0: minimum base quality (0 = very permissive, default is 13)
            # -d 10000: max per-BAM depth (increase to avoid depth filtering)
            # -C 50: adjust mapping quality (lower = more sensitive)
            # --min-MQ 0: minimum mapping quality (0 = very permissive, default is 0 but can be higher)
            # -m: multiallelic caller
            # -v: variant sites only
            # --ploidy 1: haploid (single sample)
            # -f GQ,GP: include genotype quality (optional, for better filtering later)
            with open(output_vcf, "w") as vcf_out:
                mpileup = subprocess.Popen(
                    [
                        "bcftools", "mpileup",
                        "-f", str(reference_fasta),
                        "-Q", "0",  # Minimum base quality (0 = very permissive)
                        "-d", "10000",  # Max depth per BAM (increase to avoid filtering)
                        "-C", "50",  # Adjust mapping quality (lower = more sensitive)
                        str(sorted_bam),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                subprocess.run(
                    [
                        "bcftools", "call",
                        "-mv",  # Multiallelic caller, variant sites only
                        "--ploidy", "1",
                        # Lower quality thresholds for more sensitivity
                        # Note: bcftools call doesn't have explicit quality filters in -mv mode
                        # but we can add --keep-alts to keep all alternate alleles
                    ],
                    stdin=mpileup.stdout,
                    stdout=vcf_out,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                mpileup.wait()
            
            # Cleanup intermediate files
            try:
                bam_file.unlink()
                sorted_bam.unlink()
                (sorted_bam.with_suffix(".bam.bai")).unlink()
            except Exception:
                pass
            
            return output_vcf
            
        except Exception as e:
            print(f"Warning: Variant calling failed: {e}")
            return None
    
    def _extract_chromosome_from_fasta(self, fasta_path: Path) -> str:
        """Extract chromosome name from FASTA file path or header."""
        # Try to extract from filename (e.g., chr1.fasta -> chr1)
        if 'chr' in fasta_path.name.lower():
            # Match patterns like chr1.fasta, chr21.fasta, chrX.fasta
            import re
            match = re.search(r'chr([0-9XY]+)', fasta_path.name, re.IGNORECASE)
            if match:
                return f"chr{match.group(1)}"
        
        # Fallback: try reading first header line
        try:
            with open(fasta_path, 'r') as f:
                for line in f:
                    if line.startswith('>'):
                        # Extract chromosome from header (e.g., ">chr1" or ">1")
                        header = line.strip().lstrip('>').split()[0]
                        if header.startswith('chr'):
                            return header
                        elif header.isdigit() or header in ['X', 'Y', 'M']:
                            return f"chr{header}"
                        return header
        except Exception:
            pass
        
        return None
    
    def _merge_intervals(self, intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Merge overlapping intervals into non-overlapping intervals.
        
        Args:
            intervals: List of (start, end) tuples (1-based, inclusive)
        
        Returns:
            List of merged non-overlapping intervals
        """
        if not intervals:
            return []
        
        # Sort by start position
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        
        merged = [sorted_intervals[0]]
        for current_start, current_end in sorted_intervals[1:]:
            last_start, last_end = merged[-1]
            
            # If current interval overlaps or is adjacent to last, merge them
            if current_start <= last_end + 1:  # +1 for adjacent intervals
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
        
        return merged
    
    def _extract_coverage_regions_from_sam(self, sam_file: Path, current_chromosome: str = None) -> Optional[List[Tuple[int, int]]]:
        """
        Extract actual covered intervals from SAM file by collecting read intervals and merging overlaps.
        
        This is used to filter truth variants to only those in regions actually covered by reads,
        which makes recall calculation meaningful (comparing against variants in sequenced regions,
        not the entire chromosome).
        
        CRITICAL: Uses actual covered intervals, not a min/max bounding box. A bounding box would
        include huge gaps between sparse reads, making recall artificially low.
        
        Args:
            sam_file: Path to SAM file
            current_chromosome: Chromosome name to filter by (e.g., 'chr1')
        
        Returns:
            List of (start, end) intervals (1-based, inclusive), or None if no coverage found
        """
        if not sam_file or not sam_file.exists():
            return None
        
        intervals = []
        current_chromosome_norm = self._normalize_chromosome_name(current_chromosome) if current_chromosome else None
        
        try:
            with open(sam_file, "r") as f:
                for line in f:
                    # Skip header lines
                    if line.startswith("@"):
                        continue
                    
                    fields = line.strip().split("\t")
                    if len(fields) < 11:
                        continue
                    
                    # Check if mapped (flag != 4)
                    flag = int(fields[1])
                    if flag == 4:  # Unmapped
                        continue
                    
                    # Extract chromosome and position
                    chrom = fields[2]  # Reference sequence name
                    pos = int(fields[3])  # 1-based position
                    
                    # Filter by chromosome if specified
                    if current_chromosome_norm:
                        chrom_norm = self._normalize_chromosome_name(chrom)
                        if chrom_norm != current_chromosome_norm:
                            continue
                    
                    # Extract CIGAR to compute end position
                    cigar_str = fields[5]
                    if cigar_str == '*':
                        # No CIGAR, use start position only (single base)
                        read_end = pos
                    else:
                        # Parse CIGAR to compute end position on reference
                        # pos is 1-based, so end position = pos + reference_length - 1
                        import re
                        cigar_ops = re.findall(r'(\d+)([MIDNSHPX=])', cigar_str)
                        reference_length = 0
                        for length_str, op in cigar_ops:
                            length = int(length_str)
                            if op in ['M', 'D', 'N', '=', 'X']:
                                # Operations that consume reference sequence
                                # M, =, X: match/mismatch (consume reference)
                                # D: deletion (consume reference, not query)
                                # N: skipped region (consume reference, not query)
                                reference_length += length
                            # I, S, H, P don't consume reference:
                            # I: insertion (consume query, not reference)
                            # S, H: soft/hard clip (don't consume reference)
                            # P: padding (don't consume reference)
                        
                        read_end = pos + reference_length - 1  # -1 because pos is 1-based, end is inclusive
                    
                    # Add interval (start, end) - both 1-based, inclusive
                    intervals.append((pos, read_end))
                        
        except Exception as e:
            print(f"Warning: Failed to extract coverage regions from SAM: {e}")
            return None
        
        if not intervals:
            return None
        
        # Merge overlapping intervals
        merged_intervals = self._merge_intervals(intervals)
        
        return merged_intervals
    
    def _compare_vcf_with_truth(self, query_vcf: Path, truth_vcf: Path, reference_fasta: Path, confident_regions: Path = None, current_chromosome: str = None, coverage_regions: Optional[List[Tuple[int, int]]] = None) -> Dict[str, float]:
        """
        Compare query VCF against truth VCF using Python-based comparison (hap.py optional).
        
        Args:
            coverage_regions: List of (start, end) intervals (1-based, inclusive) for filtering truth variants.
                            If None, uses all variants on the chromosome (less accurate recall).
        
        Returns:
            Dictionary with TP, FP, FN, precision, recall, F1 for SNPs and indels
        """
        if not query_vcf or not query_vcf.exists():
            return self._default_truth_metrics()
        
        if not truth_vcf or not truth_vcf.exists():
            print(f"Warning: Truth VCF not found: {truth_vcf}")
            return self._default_truth_metrics()
        
        # Try hap.py first (preferred method)
        try:
            result = subprocess.run(
                ["hap.py", "--version"],
                capture_output=True,
                check=True,
            )
            return self._compare_vcf_happy(query_vcf, truth_vcf, reference_fasta, confident_regions)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Use Python-based comparison as standard (no warning - this is the default)
            return self._compare_vcf_python(query_vcf, truth_vcf, current_chromosome=current_chromosome, coverage_regions=coverage_regions)
    
    def _compare_vcf_happy(self, query_vcf: Path, truth_vcf: Path, reference_fasta: Path, confident_regions: Path = None, current_chromosome: str = None) -> Dict[str, float]:
        """Compare VCFs using hap.py.
        
        Note: hap.py will automatically filter to chromosomes present in both files,
        so if query_vcf only contains variants for current_chromosome, the comparison
        will effectively be chromosome-specific.
        """
        output_prefix = self.config.output_dir / f"happy_{query_vcf.stem}"
        
        cmd = [
            "hap.py",
            str(truth_vcf),
            str(query_vcf),
            "-r", str(reference_fasta),
            "-o", str(output_prefix),
        ]
        
        if confident_regions and confident_regions.exists():
            cmd.extend(["-f", str(confident_regions)])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            
            # Parse hap.py summary CSV
            summary_csv = output_prefix.with_suffix(".summary.csv")
            if summary_csv.exists():
                return self._parse_happy_summary(summary_csv)
            else:
                print("Warning: hap.py summary CSV not found")
                return self._default_truth_metrics()
                
        except subprocess.CalledProcessError as e:
            print(f"Warning: hap.py failed: {e.stderr}")
            return self._default_truth_metrics()
    
    def _parse_happy_summary(self, summary_csv: Path) -> Dict[str, float]:
        """Parse hap.py summary CSV to extract metrics."""
        import csv
        
        # Initialize defaults
        tp_snp = 0.0
        fp_snp = 0.0
        fn_snp = 0.0
        tp_indel = 0.0
        fp_indel = 0.0
        fn_indel = 0.0
        
        try:
            with open(summary_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Look for SNP and INDEL rows
                    if row.get("Type") == "SNP":
                        tp_snp = float(row.get("TP", 0))
                        fp_snp = float(row.get("FP", 0))
                        fn_snp = float(row.get("FN", 0))
                    elif row.get("Type") == "INDEL":
                        tp_indel = float(row.get("TP", 0))
                        fp_indel = float(row.get("FP", 0))
                        fn_indel = float(row.get("FN", 0))
            
            # Calculate precision, recall, F1
            tp_total = tp_snp + tp_indel
            fp_total = fp_snp + fp_indel
            fn_total = fn_snp + fn_indel
            
            precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
            recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                "true_snps_tp": tp_snp,
                "true_indels_tp": tp_indel,
                "false_positives": fp_total,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
            }
        except Exception as e:
            print(f"Warning: Failed to parse hap.py summary: {e}")
            return self._default_truth_metrics()
    
    def _normalize_chromosome_name(self, chrom: str) -> str:
        """Normalize chromosome name to 'chr1', 'chr2', etc. format.
        
        Handles: '1' -> 'chr1', 'chr1' -> 'chr1', 'CHR1' -> 'chr1'
        """
        chrom = str(chrom).strip()
        chrom_lower = chrom.lower()
        if chrom_lower.startswith('chr'):
            # Already has 'chr' prefix, just normalize case
            return chrom_lower
        else:
            # Add 'chr' prefix
            return f"chr{chrom}"
    
    def _position_in_intervals(self, pos: int, intervals: List[Tuple[int, int]]) -> bool:
        """
        Check if a position (1-based) is within any of the given intervals.
        
        Args:
            pos: Position (1-based)
            intervals: List of (start, end) tuples (1-based, inclusive)
        
        Returns:
            True if position is within any interval
        """
        for start, end in intervals:
            if start <= pos <= end:
                return True
        return False
    
    def _compare_vcf_python(self, query_vcf: Path, truth_vcf: Path, current_chromosome: str = None, coverage_regions: Optional[List[Tuple[int, int]]] = None) -> Dict[str, float]:
        """Python-based VCF comparison (standard method).
        
        IMPORTANT: Filters truth variants by:
        1. Current chromosome (to avoid comparing against all chromosomes)
        2. Coverage regions (to only count variants in regions actually sequenced)
        
        Without region filtering, recall is artificially near-zero because the denominator
        includes all truth variants on the chromosome, even in regions never sequenced.
        
        Args:
            coverage_regions: List of (start, end) intervals (1-based, inclusive) for filtering truth variants.
                            If None, uses all variants on chromosome (less accurate).
        """
        # Use cached truth VCF variants for fast comparison
        if self._truth_vcf_cache is not None:
            all_truth_variants = self._truth_vcf_cache
        else:
            # Fallback: parse truth VCF (slow, but only if cache not available)
            all_truth_variants = self._parse_vcf_variants(truth_vcf)
        
        # CRITICAL FIX: Filter truth variants to current chromosome only
        # This ensures recall denominator is correct (not using all chromosomes 1-22)
        if current_chromosome:
            # Normalize chromosome name for comparison
            current_chromosome_norm = self._normalize_chromosome_name(current_chromosome)
            
            # Filter truth variants: normalize both sides for comparison
            truth_variants = {
                key: var for key, var in all_truth_variants.items()
                if self._normalize_chromosome_name(var["chrom"]) == current_chromosome_norm
            }
        else:
            # If no chromosome specified, use all (shouldn't happen in normal operation)
            truth_variants = all_truth_variants
            print("WARNING: No current_chromosome specified - using ALL truth variants (recall will be wrong!)")
        
        # CRITICAL FIX: Filter truth variants by actual covered intervals (not bounding box)
        # This ensures recall denominator only includes variants in regions actually sequenced
        # Without this, recall is ~0.0002 because denominator includes 300k+ variants on entire chromosome
        # but reads only cover a tiny fraction (e.g., 0.1-0.5% of chromosome)
        # 
        # Using actual intervals (not min/max bounding box) is essential: sparse reads create huge gaps,
        # and a bounding box would include all variants in those gaps, making recall artificially low.
        truth_variants_before_region_filter = len(truth_variants)
        
        if coverage_regions and len(coverage_regions) > 0:
            # Filter truth variants to positions within any covered interval
            # Note: VCF positions are 1-based, same as SAM intervals
            truth_variants = {
                key: var for key, var in truth_variants.items()
                if self._position_in_intervals(var["pos"], coverage_regions)
            }
            truth_variants_after_region_filter = len(truth_variants)
            
            # Calculate total covered span
            total_covered_span = sum(end - start + 1 for start, end in coverage_regions)
            
            # Debug logging (only first few times)
            if not hasattr(self, '_debug_region_filter_count'):
                self._debug_region_filter_count = 0
            if self._debug_region_filter_count < 3 and current_chromosome:
                num_intervals = len(coverage_regions)
                if num_intervals <= 5:
                    intervals_str = ", ".join([f"{s:,}-{e:,}" for s, e in coverage_regions])
                else:
                    intervals_str = f"{num_intervals} intervals (first: {coverage_regions[0][0]:,}-{coverage_regions[0][1]:,}, last: {coverage_regions[-1][0]:,}-{coverage_regions[-1][1]:,})"
                
                print(f"DEBUG Region filtering for {current_chromosome_norm if current_chromosome else 'unknown'}:")
                print(f"  Coverage intervals: {intervals_str}")
                print(f"  Total covered span: {total_covered_span:,} bp ({num_intervals} interval{'s' if num_intervals != 1 else ''})")
                print(f"  Truth variants (chr-filtered): {truth_variants_before_region_filter:,}")
                print(f"  Truth variants (chr+interval-filtered): {truth_variants_after_region_filter:,}")
                print(f"  Reduction: {truth_variants_before_region_filter - truth_variants_after_region_filter:,} variants excluded")
                self._debug_region_filter_count += 1
        else:
            # No region filtering - warn that recall will be artificially low
            if not hasattr(self, '_warned_no_region_filter'):
                print("WARNING: No coverage regions provided - recall will be artificially low!")
                print("  (Truth variants not filtered by read coverage regions)")
                self._warned_no_region_filter = True
        
        # CRITICAL: Handle chromosomes with no truth data (e.g., chrX in HG002 v4.2.1)
        # If there are no truth variants for this chromosome, return default metrics
        # and skip truth-based reward contribution
        if len(truth_variants) == 0:
            if not hasattr(self, '_warned_no_truth_data'):
                print(f"WARNING: No truth variants found for {current_chromosome_norm if current_chromosome else 'unknown'}")
                print("  (HG002 v4.2.1 only includes autosomes 1-22, not chrX/chrY)")
                print("  Truth-based metrics will be set to 0 for this episode")
                self._warned_no_truth_data = True
            # Return default metrics (all zeros) - reward will skip truth metrics
            return self._default_truth_metrics()
        
        query_variants_list = list(self._parse_vcf_variants(query_vcf).values())
        
        # Filter query variants to current chromosome as well (for consistency)
        if current_chromosome:
            current_chromosome_norm = self._normalize_chromosome_name(current_chromosome)
            query_variants_list = [
                q_var for q_var in query_variants_list
                if self._normalize_chromosome_name(q_var["chrom"]) == current_chromosome_norm
            ]
        
        # CRITICAL FIX: Normalize chromosome names in keys for proper matching
        # Truth VCF might have "1", "2" while query VCF has "chr1", "chr2" - normalize both
        def normalize_key(key):
            """Normalize variant key to use consistent chromosome naming."""
            chrom, pos, ref, alt = key
            chrom_norm = self._normalize_chromosome_name(chrom)
            return (chrom_norm, pos, ref, alt)
        
        # Create normalized sets for comparison
        query_keys_normalized = {normalize_key((q["chrom"], q["pos"], q["ref"], q["alt"])) 
                                 for q in query_variants_list}
        truth_keys_normalized = {normalize_key(key) for key in truth_variants.keys()}
        
        # Count matches (TP) - use normalized keys
        tp_snp = 0
        tp_indel = 0
        fp = 0
        
        for q_var in query_variants_list:
            key_norm = normalize_key((q_var["chrom"], q_var["pos"], q_var["ref"], q_var["alt"]))
            var_type = "INDEL" if (len(q_var["ref"]) != 1 or len(q_var["alt"]) != 1) else "SNP"
            
            if key_norm in truth_keys_normalized:
                if var_type == "SNP":
                    tp_snp += 1
                else:
                    tp_indel += 1
            else:
                fp += 1
        
        # Count false negatives (only for current chromosome) - use normalized keys
        fn_snp = sum(1 for key, t_var in truth_variants.items() 
                    if t_var["type"] == "SNP" and normalize_key(key) not in query_keys_normalized)
        fn_indel = sum(1 for key, t_var in truth_variants.items() 
                      if t_var["type"] == "INDEL" and normalize_key(key) not in query_keys_normalized)
        
        tp_total = tp_snp + tp_indel
        fp_total = fp
        fn_total = fn_snp + fn_indel
        
        # CRITICAL: Verify denominator is chromosome+region-specific
        # Recall = TP / (TP + FN) where FN counts only truth variants for current chromosome AND covered regions
        # Denominator should be: TP + FN = total truth variants for current chromosome in covered regions
        total_truth_variants_filtered = len(truth_variants)  # This is the correct denominator (chr + region filtered)
        recall_denominator = tp_total + fn_total  # Should equal total_truth_variants_filtered
        
        # Debug: verify denominator is correct (only print first few times to avoid spam)
        if not hasattr(self, '_debug_recall_count'):
            self._debug_recall_count = 0
        if self._debug_recall_count < 3 and current_chromosome:
            region_info = ""
            if coverage_regions and len(coverage_regions) > 0:
                if len(coverage_regions) == 1:
                    region_info = f" (interval: {coverage_regions[0][0]:,}-{coverage_regions[0][1]:,})"
                else:
                    region_info = f" ({len(coverage_regions)} intervals, span: {sum(e-s+1 for s,e in coverage_regions):,} bp)"
            print(f"DEBUG Recall calculation for {current_chromosome_norm}{region_info}:")
            print(f"  Total truth variants (chr+interval-filtered): {total_truth_variants_filtered:,}")
            print(f"  TP: {tp_total}, FN: {fn_total}, Recall denominator: {recall_denominator:,}")
            recall_value = tp_total / recall_denominator if recall_denominator > 0 else 0.0
            print(f"  Recall = {tp_total} / {recall_denominator:,} = {recall_value:.6f}")
            if abs(recall_denominator - total_truth_variants_filtered) > 1:
                print(f"  WARNING: Denominator mismatch! Expected {total_truth_variants_filtered:,}, got {recall_denominator:,}")
            self._debug_recall_count += 1
        
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "true_snps_tp": float(tp_snp),
            "true_indels_tp": float(tp_indel),
            "false_positives": float(fp_total),
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
        }
    
    def _parse_vcf_variants(self, vcf_file: Path) -> Dict:
        """Parse VCF file and return dictionary of variants. Handles both plain and gzipped files."""
        import gzip
        
        variants = {}
        
        try:
            # Check if file is gzipped by extension or by reading first bytes
            is_gzipped = False
            if vcf_file.suffix == ".gz":
                is_gzipped = True
            else:
                # Check magic bytes for gzip (0x1f 0x8b)
                try:
                    with open(vcf_file, "rb") as test_f:
                        magic = test_f.read(2)
                        if magic == b'\x1f\x8b':
                            is_gzipped = True
                except Exception:
                    pass
            
            # Open file with appropriate method
            if is_gzipped:
                open_func = gzip.open
                mode = "rt"  # text mode for gzip
            else:
                open_func = open
                mode = "r"
            
            with open_func(vcf_file, mode) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    
                    fields = line.strip().split("\t")
                    if len(fields) < 5:
                        continue
                    
                    chrom = fields[0]
                    pos = int(fields[1])
                    ref = fields[3]
                    alt = fields[4]
                    
                    # Skip multi-allelic for simplicity
                    if "," in alt:
                        continue
                    
                    var_type = "INDEL" if (len(ref) != 1 or len(alt) != 1) else "SNP"
                    key = (chrom, pos, ref, alt)
                    variants[key] = {
                        "chrom": chrom,
                        "pos": pos,
                        "ref": ref,
                        "alt": alt,
                        "type": var_type,
                    }
        except Exception as e:
            print(f"Warning: Failed to parse VCF {vcf_file}: {e}")
        
        return variants
    
    def _default_truth_metrics(self) -> Dict[str, float]:
        """Return default truth-based metrics."""
        return {
            "true_snps_tp": 0.0,
            "true_indels_tp": 0.0,
            "false_positives": 0.0,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
    
    def _parse_sam_metrics(self, sam_file: Path) -> Dict[str, float]:
        """
        Parse SAM file to extract mapping metrics and variant detection.
        
        Returns:
            Dictionary with mapping_rate, alignment_score, edit_distance,
            indel_count, snp_count, total_variants
        """
        if not sam_file.exists():
            return self._default_metrics()
        
        try:
            with open(sam_file, "r") as f:
                lines = f.readlines()
            
            # Filter out header lines
            data_lines = [l for l in lines if not l.startswith("@")]
            
            if not data_lines:
                return self._default_metrics()
            
            total_reads = len(data_lines)
            mapped_reads = 0
            alignment_scores = []
            edit_distances = []
            total_indels = 0
            total_snps = 0
            
            for line in data_lines:
                fields = line.strip().split("\t")
                if len(fields) < 11:
                    continue
                
                # Check if mapped (flag != 4)
                flag = int(fields[1])
                if flag != 4:  # Not unmapped
                    mapped_reads += 1
                    
                    # Extract CIGAR string (field 5)
                    cigar_str = fields[5]
                    
                    # Extract alignment score (AS tag)
                    as_score = None
                    for field in fields[11:]:
                        if field.startswith("AS:i:"):
                            as_score = int(field.split(":")[2])
                            alignment_scores.append(as_score)
                            break
                    
                    # Extract edit distance (NM tag)
                    nm_dist = None
                    md_tag = None
                    for field in fields[11:]:
                        if field.startswith("NM:i:"):
                            nm_dist = int(field.split(":")[2])
                            edit_distances.append(nm_dist)
                        elif field.startswith("MD:Z:"):
                            md_tag = field
                    
                    # Parse variants from CIGAR
                    if cigar_str != '*':
                        variants = self._parse_cigar_variants(cigar_str)
                        indel_events = variants["indel_count"]
                        total_indels += indel_events
                        
                        # Count SNPs: use MD tag if available, otherwise estimate from edit distance
                        if md_tag:
                            snps = self._parse_md_tag_for_snps(md_tag, cigar_str)
                            total_snps += snps
                        else:
                            # Estimate SNPs from edit distance (NM tag)
                            # Edit distance = SNPs + indels (in bases)
                            # We need to count indel bases, not just events
                            if nm_dist is not None:
                                # Count total indel bases from CIGAR
                                indel_bases = self._count_indel_bases(cigar_str)
                                # SNPs = edit_distance - indel_bases
                                estimated_snps = max(0, nm_dist - indel_bases)
                                total_snps += estimated_snps
            
            mapping_rate = mapped_reads / total_reads if total_reads > 0 else 0.0
            avg_alignment_score = np.mean(alignment_scores) if alignment_scores else 0.0
            avg_edit_distance = np.mean(edit_distances) if edit_distances else 0.0
            
            metrics = {
                "mapping_rate": mapping_rate,
                "alignment_score": avg_alignment_score,
                "edit_distance": avg_edit_distance,
            }
            
            # Use truth-based metrics if enabled, otherwise use raw counts
            if self.config.use_truth_metrics:
                # Truth-based metrics will be added in _run_gdiet after variant calling
                # For now, keep raw counts as fallback
                metrics.update({
                    "indel_count": float(total_indels),
                    "snp_count": float(total_snps),
                    "total_variants": float(total_indels + total_snps),
                })
            else:
                # Use raw counts
                metrics.update({
                    "indel_count": float(total_indels),
                    "snp_count": float(total_snps),
                    "total_variants": float(total_indels + total_snps),
                })
            
            return metrics
        except Exception as e:
            print(f"Warning: Failed to parse SAM file {sam_file}: {e}")
            return self._default_metrics()

    def _default_metrics(self) -> Dict[str, float]:
        """Return default metrics when parsing fails."""
        metrics = {
            "mapping_rate": 0.0,
            "alignment_score": 0.0,
            "edit_distance": 1000.0,  # High penalty
            "indel_count": 0.0,
            "snp_count": 0.0,
            "total_variants": 0.0,
        }
        # Add truth-based metrics if enabled
        if self.config.use_truth_metrics:
            metrics.update({
                "true_snps_tp": 0.0,
                "true_indels_tp": 0.0,
                "false_positives": 0.0,
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            })
        return metrics

    def _compute_reward(
        self, runtime: float, metrics: Dict[str, float], pattern_str: str = None
    ) -> float:
        """Compute reward from runtime, accuracy metrics, and variant detection.
        
        Reward structure:
        1. Soft trade-offs (normalized):
           - Truth metrics (F1, indels, FP penalty) - PRIMARY accuracy measures
           - Runtime weight - balanced with accuracy
           - Mapping rate - prevents low-sensitivity patterns
           - Edit distance - soft cap approach (ED <= 3: small penalty, ED > 3: increasing penalty)
        
        2. Hard constraints (applied after normalization):
           - mapping_rate < 0.9 → big penalty (episode failed)
           - edit_distance > 10 → big penalty (alignment garbage)
        
        Edit Distance Soft Cap:
        - ED <= 3: small or no penalty (good enough for Illumina)
        - ED > 3: quickly increasing penalty (prevents bad alignments)
        
        IMPORTANT: When truth metrics are available (use_truth_metrics=True), ONLY truth-based
        metrics are used. Raw variant counts (snp_count_weight, indel_count_weight, etc.) are
        NOT included to avoid rewarding false positives. Raw counts incentivize calling more
        variants regardless of correctness, which conflicts with precision/recall objectives.
        
        For small datasets: runtime weight is reduced and accuracy weights are increased
        (indirectly encourages less sparsification by making accuracy more valuable)
        
        All weights are normalized to ensure balanced contribution.
        """
        # Dataset-size-aware reward adjustment (indirect approach)
        # For small datasets: reduce runtime importance, increase accuracy importance
        # This indirectly encourages less sparsification by making accuracy more valuable than speed
        if self._dataset_features:
            num_reads_normalized = self._dataset_features.get("num_reads", 0.0)
            
            if num_reads_normalized < self.config.small_dataset_threshold:
                # Small dataset: reduce runtime weight, increase accuracy weights
                runtime_weight = self.config.runtime_weight * self.config.small_dataset_runtime_weight_multiplier
                mapping_rate_weight = self.config.mapping_rate_weight * self.config.small_dataset_accuracy_weight_multiplier
                alignment_score_weight = self.config.alignment_score_weight * self.config.small_dataset_accuracy_weight_multiplier
                edit_distance_weight = self.config.edit_distance_weight * self.config.small_dataset_accuracy_weight_multiplier
            else:
                # Normal dataset: use standard weights
                runtime_weight = self.config.runtime_weight
                mapping_rate_weight = self.config.mapping_rate_weight
                alignment_score_weight = self.config.alignment_score_weight
                edit_distance_weight = self.config.edit_distance_weight
        else:
            # Fallback: use standard weights
            runtime_weight = self.config.runtime_weight
            mapping_rate_weight = self.config.mapping_rate_weight
            alignment_score_weight = self.config.alignment_score_weight
            edit_distance_weight = self.config.edit_distance_weight
        
        # Compute edit distance penalty with soft cap approach
        # ED <= 3: small/no penalty (good enough for Illumina)
        # ED > 3: quickly increasing penalty
        edit_distance = metrics["edit_distance"]
        
        if edit_distance <= self.config.edit_distance_soft_cap:
            # Soft region: ED <= 3, small or no penalty
            edit_distance_penalty = edit_distance_weight * edit_distance
        else:
            # Hard region: ED > 3, quickly increasing penalty
            # Base penalty for being above soft cap
            edit_distance_penalty = edit_distance_weight * self.config.edit_distance_soft_cap
            # Extra penalty for excess above soft cap (with multiplier)
            excess_edit_distance = edit_distance - self.config.edit_distance_soft_cap
            extra_penalty = (
                self.config.edit_distance_high_penalty_multiplier 
                * abs(edit_distance_weight)  # Use absolute value for penalty direction
                * excess_edit_distance
            )
            edit_distance_penalty -= extra_penalty  # Subtract because weight is already negative
        
        # Base reward components
        reward = (
            runtime_weight * runtime
            + mapping_rate_weight * metrics["mapping_rate"]
            + alignment_score_weight * metrics["alignment_score"]
            + edit_distance_penalty
        )
        
        # Use truth-based metrics if available, otherwise use raw counts
        # IMPORTANT: When truth metrics are available, DO NOT use raw counts to avoid
        # rewarding false positives (raw counts incentivize calling more variants regardless of correctness)
        if self.config.use_truth_metrics and "true_snps_tp" in metrics:
            # CRITICAL: Scale truth metrics to make them meaningful
            # F1 and recall are typically 0.0006-0.0007 (tiny), so they need scaling
            # to provide meaningful gradients for learning
            # Scale F1 by 1000: 0.0006 → 0.6, 0.0007 → 0.7 (meaningful difference)
            # Scale recall by 1000: 0.0003 → 0.3 (more informative)
            # TP counts are already in reasonable range (50-100), keep as-is
            # FP counts are also in reasonable range (200-300), keep as-is
            
            f1_score = metrics.get("f1_score", 0.0)
            recall = metrics.get("recall", 0.0)
            precision = metrics.get("precision", 0.0)
            
            # Scale F1 and recall to make them meaningful (multiply by 1000)
            # This makes 0.0006 → 0.6, which is comparable to other reward components
            f1_scaled = f1_score * 1000.0
            recall_scaled = recall * 1000.0
            precision_scaled = precision * 1000.0
            
            # Use scaled metrics in reward
            # F1 is the main metric (already combines precision and recall)
            # But also include precision and recall separately for more signal
            reward += (
                self.config.true_snps_tp_weight * metrics.get("true_snps_tp", 0.0)
                + self.config.true_indels_tp_weight * metrics.get("true_indels_tp", 0.0)
                + self.config.f1_score_weight * f1_scaled  # Scaled F1
                + self.config.false_positives_penalty * metrics.get("false_positives", 0.0)
                # Add precision and recall as additional signals (with smaller weights)
                + 0.5 * precision_scaled  # Precision signal (scaled)
                + 0.5 * recall_scaled  # Recall signal (scaled)
            )
            # Explicitly DO NOT add raw count rewards when truth metrics are available
            # This prevents incentivizing false positives
        else:
            # Raw variant counts (fallback when truth metrics unavailable)
            reward += (
                self.config.snp_count_weight * metrics.get("snp_count", 0.0)
                + self.config.indel_count_weight * metrics.get("indel_count", 0.0)
                + self.config.total_variants_weight * metrics.get("total_variants", 0.0)
            )
        
        # Calculate normalization factor if enabled (includes all weights and penalty multiplier)
        normalization_factor = 1.0
        if self.config.normalize_weights:
            # Collect all weight magnitudes (absolute values)
            weight_magnitudes = [
                abs(runtime_weight),
                abs(mapping_rate_weight),
                abs(alignment_score_weight),
                abs(edit_distance_weight),
            ]
            
            # Add variant detection weights
            # IMPORTANT: When truth metrics are available, use ONLY truth-based weights
            # Raw count weights should NOT be included to avoid rewarding false positives
            if self.config.use_truth_metrics and "true_snps_tp" in metrics:
                weight_magnitudes.extend([
                    abs(self.config.true_snps_tp_weight),
                    abs(self.config.true_indels_tp_weight),
                    abs(self.config.f1_score_weight),
                    abs(self.config.false_positives_penalty),
                    # Include precision and recall weights (0.5 each, scaled by 1000)
                    # These are hardcoded in the reward function, so include them here
                    0.5,  # precision_scaled weight
                    0.5,  # recall_scaled weight
                ])
                # Explicitly DO NOT include raw count weights when truth metrics are available
            else:
                # Only use raw count weights when truth metrics are NOT available
                weight_magnitudes.extend([
                    abs(self.config.snp_count_weight),
                    abs(self.config.indel_count_weight),
                    abs(self.config.total_variants_weight),
                ])
            
            # Include mapping rate threshold penalty multiplier in normalization
            # This ensures the penalty is also normalized
            mapping_rate_penalty_multiplier = 5.0  # Fixed penalty multiplier
            weight_magnitudes.append(mapping_rate_penalty_multiplier)
            
            # Calculate normalization factor (target sum of absolute weights)
            # Target: sum of absolute weights = 20 (reasonable scale for rewards)
            current_sum = sum(weight_magnitudes)
            if current_sum > 0:
                normalization_factor = 20.0 / current_sum
        
        # Apply normalization to reward
        reward = reward * normalization_factor
        
        # Hard constraints: big penalties for failed episodes
        # These are applied AFTER normalization to ensure they have strong impact
        
        # Hard threshold: mapping rate < 0.9 → episode failed
        if metrics["mapping_rate"] < self.config.mapping_rate_hard_threshold:
            reward += self.config.mapping_rate_hard_penalty
        
        # Hard threshold: edit distance > 10 → alignment garbage
        if edit_distance > self.config.edit_distance_hard_threshold:
            reward += self.config.edit_distance_hard_penalty
        
        # Soft penalty: mapping rate below soft threshold (gradual penalty)
        if metrics["mapping_rate"] < self.config.min_mapping_rate_threshold:
            penalty_multiplier = 5.0 * normalization_factor
            penalty = -penalty_multiplier * (self.config.min_mapping_rate_threshold - metrics["mapping_rate"])
            reward += penalty
        
        return float(reward)

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        self._episode_count += 1
        
        # Select dataset for this episode (multi-dataset mode)
        if self.config.dataset_manager is not None:
            self._current_reference_fasta, self._current_query_fastq, self._current_chromosome = (
                self.config.dataset_manager.get_random_dataset()
            )
        
        # Compute dataset features for current dataset
        self._dataset_features = self._extract_dataset_features(
            reference_fasta=self._current_reference_fasta,
            query_fastq=self._current_query_fastq,
        )
        
        # Initialize observation with ONLY dataset features
        # Observation: [num_reads, avg_read_length, reference_size, gc_content]
        # These are inputs the agent uses to decide which pattern to select
        # Performance metrics are outputs (used for reward, not in observation)
        self._current_observation = np.array(
            [
                self._dataset_features["num_reads"],
                self._dataset_features["avg_read_length"],
                self._dataset_features["reference_size"],
                self._dataset_features["gc_content"],
            ],
            dtype=np.float32,
        )
        
        info = {
            "episode": self._episode_count,
            "dataset_features": self._dataset_features.copy(),
            "chromosome": self._current_chromosome,
            "reference_fasta": str(self._current_reference_fasta),
        }
        return self._current_observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step: run GDiet with generated pattern and return metrics.
        
        Args:
            action: Array of binary decisions (0=skip, 1=keep) for each position
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert to numpy array if needed
        action = np.asarray(action, dtype=np.int32)
        assert self.action_space.contains(action), f"Action {action} out of bounds"
        
        # Run GDiet with selected pattern
        runtime, metrics = self._run_gdiet(action)
        
        # Get pattern string for reward computation
        pattern_str = self._action_to_pattern_string(action)
        
        # Ensure dataset features are computed (should already be set in reset)
        if self._dataset_features is None:
            self._dataset_features = self._extract_dataset_features(
                reference_fasta=self._current_reference_fasta,
                query_fastq=self._current_query_fastq,
            )
        
        # Observation remains the same (dataset features only)
        # Performance metrics are outputs used for reward, not part of observation
        # The agent doesn't need to see results to make the next decision (single-step episodes)
        self._current_observation = np.array(
            [
                self._dataset_features["num_reads"],
                self._dataset_features["avg_read_length"],
                self._dataset_features["reference_size"],
                self._dataset_features["gc_content"],
            ],
            dtype=np.float32,
        )
        
        # Compute reward (pass pattern_str for dataset-size-aware penalty)
        reward = self._compute_reward(runtime, metrics, pattern_str=pattern_str)
        
        # This is a single-step environment (one action = one episode)
        # So we always terminate after one step
        terminated = True
        truncated = False
        
        info = {
            "pattern": self._action_to_pattern_string(action),
            "runtime": runtime,
            "chromosome": self._current_chromosome,
            **metrics,
            "reward": reward,
        }
        
        return self._current_observation, reward, terminated, truncated, info

    def render(self) -> None:
        """Render the current state."""
        print(
            f"GenomeDietPatternEnv | "
            f"episode={self._episode_count} | "
            f"obs={self._current_observation} | "
            f"pattern_length={self.config.pattern_length}"
        )

    def close(self) -> None:
        """Clean up resources."""
        pass


def load_env_from_config(
    config_dict: Dict[str, Any], project_root: Path | None = None
) -> GenomeDietPatternEnv:
    """
    Helper to instantiate the environment from a raw dictionary.
    
    Args:
        config_dict: Configuration dictionary (typically from YAML)
                    Should have 'env' key with environment configuration
        project_root: Root directory of the project (for resolving relative paths)
    """
    # Extract env config
    if "env" not in config_dict:
        raise ValueError("config_dict must have 'env' key")
    
    env_cfg = config_dict["env"].copy()
    
    # Handle multi-dataset mode: initialize dataset manager if chromosomes_dir is provided
    if "chromosomes_dir" in env_cfg and env_cfg["chromosomes_dir"]:
        # Multi-dataset mode
        # Resolve paths
        if project_root is None:
            current = Path.cwd()
            if (current / "GDiet-ShortReads").exists():
                project_root = current
            elif (current.parent / "GDiet-ShortReads").exists():
                project_root = current.parent
            else:
                project_root = current
        
        chromosomes_dir = Path(env_cfg["chromosomes_dir"])
        if not chromosomes_dir.is_absolute():
            chromosomes_dir = (project_root / chromosomes_dir).resolve()
        
        # Handle chromosome-specific FASTQ files (preferred) or shared FASTQ (legacy)
        query_fastq_dir = env_cfg.get("query_fastq_dir")
        query_fastq = env_cfg.get("query_fastq")
        
        resolved_query_fastq_dir = None
        resolved_query_fastq = None
        
        if query_fastq_dir:
            # Chromosome-specific FASTQ files mode
            resolved_query_fastq_dir = Path(query_fastq_dir)
            if not resolved_query_fastq_dir.is_absolute():
                resolved_query_fastq_dir = (project_root / resolved_query_fastq_dir).resolve()
        elif query_fastq:
            # Legacy: shared FASTQ file
            resolved_query_fastq = Path(query_fastq)
            if not resolved_query_fastq.is_absolute():
                resolved_query_fastq = (project_root / resolved_query_fastq).resolve()
        else:
            raise ValueError("Either query_fastq_dir or query_fastq must be provided in config")
        
        # Initialize dataset manager
        chromosomes = env_cfg.get("chromosomes", None)
        dataset_manager = ChromosomeDatasetManager(
            chromosomes_dir=chromosomes_dir,
            query_fastq=resolved_query_fastq,
            query_fastq_dir=resolved_query_fastq_dir,
            chromosomes=chromosomes,
        )
        env_cfg["dataset_manager"] = dataset_manager
    
    # Create config from env_cfg (not full config_dict)
    config = GenomeDietEnvConfig.from_dict(env_cfg, project_root=project_root)
    return GenomeDietPatternEnv(config)
