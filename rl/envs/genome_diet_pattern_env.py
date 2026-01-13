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
    runtime_normalization_factor: float = 2.0  # Divide runtime by this factor for normalization (2.0 = 1.72s → 0.86)
    mapping_rate_weight: float = 1.5  # Moderate priority: prevent low-sensitivity patterns
    alignment_score_weight: float = 0.0  # Disabled: truth metrics are primary accuracy measures
    edit_distance_weight: float = -0.5  # Low continuous influence: soft cap approach
    edit_distance_soft_cap: float = 3.0  # ED <= 3: small/no penalty (good enough for Illumina)
    edit_distance_high_threshold: float = 3.0  # Threshold for "high" edit distance (moved from 20 to 3)
    edit_distance_high_penalty_multiplier: float = 4.0  # Strong extra penalty above soft cap
    edit_distance_hard_threshold: float = 10.0  # Hard threshold: ED > 10 gets big penalty (alignment garbage)
    edit_distance_critical_threshold: float = 11.0  # Critical threshold: ED > 11 gets very strong penalty
    edit_distance_critical_penalty_multiplier: float = 5.0  # Strong multiplier for ED > 11
    snp_count_weight: float = 0.0
    indel_count_weight: float = 0.0
    total_variants_weight: float = 0.0
    min_mapping_rate_threshold: float = 0.5  # Soft penalty if mapping rate below this threshold
    mapping_rate_hard_threshold: float = 0.9  # Hard threshold: mapping_rate < 0.9 gets big penalty (episode failed)
    mapping_rate_hard_penalty: float = -3.0  # Big penalty for mapping rate below hard threshold
    edit_distance_hard_penalty: float = -3.0  # Big penalty for edit distance above hard threshold
    
    true_snps_tp_weight: float = 0.0
    true_indels_tp_weight: float = 0.8
    f1_score_weight: float = 4.0
    false_positives_penalty: float = -1.0
    normalize_weights: bool = True  # Whether to normalize all weights to sum to a target magnitude
    
    small_dataset_runtime_weight_multiplier: float = 0.3
    small_dataset_accuracy_weight_multiplier: float = 1.5
    small_dataset_threshold: float = 0.01
    observation_dim: int = 4
    output_dir: Path = Path("rl/outputs")
    
    truth_vcf: Path = None
    confident_regions_bed: Path = None
    use_truth_metrics: bool = False
    
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
        self._evaluation_mode = False  # Flag to force variant calling every episode during evaluation
        
        # Track reward component contributions for analysis
        self._reward_contributions = []  # List of dicts with component contributions
        self._log_contributions_freq = 10  # Log contribution stats every N episodes
        
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
        
        self.action_space = spaces.MultiDiscrete([2] * self.config.pattern_length)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.observation_dim,),
            dtype=np.float32,
        )

    def _action_to_pattern_string(self, action: np.ndarray) -> str:
        """Convert action array to pattern string.
        
        The rightmost '1' marks where the pattern ends.
        Pattern includes positions from 0 up to (but not including) that rightmost '1'.
        """
        # Action is array of 0s and 1s, convert to string
        pattern_list = [str(int(a)) for a in action]
        
        last_one_idx = -1
        for i in range(len(pattern_list) - 1, -1, -1):
            if pattern_list[i] == '1':
                last_one_idx = i
                break
        
        if last_one_idx == -1:
            return "1"
        
        if last_one_idx == 0:
            return "1"
        
        pattern = ''.join(pattern_list[:last_one_idx])
        
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
        
        query_fastq = query_fastq or self._current_query_fastq or self.config.query_fastq
        reference_fasta = reference_fasta or self._current_reference_fasta or self.config.reference_fasta
        
        try:
            with open(query_fastq, "r") as f:
                lines = f.readlines()
            
            sequence_lines = [line.strip() for i, line in enumerate(lines) if i % 4 == 1]
            num_reads = len(sequence_lines)
            
            read_lengths = []
            gc_counts = {"G": 0, "C": 0, "A": 0, "T": 0, "N": 0, "total": 0}
            
            for seq in sequence_lines:
                seq = seq.upper()
                read_lengths.append(len(seq))
                for base in seq:
                    if base in gc_counts:
                        gc_counts[base] += 1
                        gc_counts["total"] += 1
            
            avg_read_length = np.mean(read_lengths) if read_lengths else 0.0
            
            if gc_counts["total"] > 0:
                gc_content = (gc_counts["G"] + gc_counts["C"]) / gc_counts["total"]
            else:
                gc_content = 0.0
            
            features["num_reads"] = num_reads / 10000.0
            features["avg_read_length"] = avg_read_length / 1000.0
            features["gc_content"] = gc_content
            
        except Exception as e:
            print(f"Warning: Failed to extract FASTQ features: {e}")
            features["num_reads"] = 0.0
            features["avg_read_length"] = 0.0
            features["gc_content"] = 0.0
        
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
                
                features["reference_size"] = total_bases / 1e9
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
        output_sam = self.config.output_dir / f"gdiet_pattern_{pattern_str}_ep{self._episode_count}.sam"
        reference_fasta = self._current_reference_fasta or self.config.reference_fasta
        query_fastq = self._current_query_fastq or self.config.query_fastq
        
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
            "--MD",
            "-a",
            "-o", str(output_sam),
            str(reference_fasta),
            str(query_fastq),
        ]
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600,
            )
            runtime = time.time() - start_time
        except subprocess.TimeoutExpired:
            runtime = 3600.0
            metrics = self._default_metrics()
            return runtime, metrics
        except subprocess.CalledProcessError as e:
            runtime = time.time() - start_time
            print(f"Warning: GDiet failed with pattern {pattern_str}: {e.stderr}")
            if output_sam.exists():
                try:
                    output_sam.unlink()
                except Exception:
                    pass
            metrics = self._default_metrics()
            return runtime, metrics
        
        if not output_sam.exists():
            print(f"Warning: SAM file not created: {output_sam}")
            metrics = self._default_metrics()
            return runtime, metrics
        
        if output_sam.stat().st_size == 0:
            print(f"Warning: SAM file is empty: {output_sam}")
            try:
                output_sam.unlink()
            except Exception:
                pass
            metrics = self._default_metrics()
            return runtime, metrics
        
        metrics = self._parse_sam_metrics(output_sam)
        
        if self.config.use_truth_metrics and self.config.truth_vcf:
            if self._evaluation_mode:
                run_variant_calling = True
            else:
                run_variant_calling = (self._episode_count % 5 == 0)
            
            if run_variant_calling:
                if self._evaluation_mode:
                    print(f"  [Evaluation] Running variant calling for episode {self._episode_count}...")
                query_vcf = self._call_variants_from_sam(output_sam, reference_fasta)
                if query_vcf is None:
                    pass
            else:
                query_vcf = None
            if query_vcf:
                current_chromosome = self._current_chromosome or self._extract_chromosome_from_fasta(reference_fasta)
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
                
                try:
                    query_vcf.unlink()
                    bam_pattern = output_sam.stem
                    for bam_file in self.config.output_dir.glob(f"{bam_pattern}*.bam*"):
                        try:
                            bam_file.unlink()
                        except Exception:
                            pass
                except Exception:
                    pass
        
        try:
            output_sam.unlink()
        except Exception:
            pass
        
        return runtime, metrics

    def _parse_cigar_variants(self, cigar_str: str) -> Dict[str, int]:
        """Parse CIGAR string to count indels and SNPs."""
        indels = 0
        snps = 0
        
        import re
        cigar_ops = re.findall(r'(\d+)([MIDNSHPX=])', cigar_str)
        
        for length_str, op in cigar_ops:
            length = int(length_str)
            if op == 'I' or op == 'D':
                indels += 1
            elif op == 'X':
                snps += length
        
        return {"indel_count": indels, "snp_count": snps}
    
    def _count_indel_bases(self, cigar_str: str) -> int:
        """Count total bases affected by indels."""
        import re
        cigar_ops = re.findall(r'(\d+)([MIDNSHPX=])', cigar_str)
        indel_bases = 0
        
        for length_str, op in cigar_ops:
            if op == 'I' or op == 'D':
                indel_bases += int(length_str)
        
        return indel_bases
    
    def _parse_md_tag_for_snps(self, md_tag: str, cigar_str: str) -> int:
        """Parse MD tag to count SNPs."""
        if not md_tag or not md_tag.startswith("MD:Z:"):
            return 0
        
        md_str = md_tag.split(":")[2]
        snp_count = 0
        
        import re
        parts = re.split(r'\^[ACGTN]+', md_str)
        for part in parts:
            snp_count += len(re.findall(r'[ACGTN]', part))
        
        return snp_count
    
    def _call_variants_from_sam(self, sam_file: Path, reference_fasta: Path) -> Path:
        """Call variants from SAM file using bcftools."""
        output_vcf = self.config.output_dir / f"variants_{sam_file.stem}.vcf"
        
        try:
            bam_file = self.config.output_dir / f"{sam_file.stem}.bam"
            sorted_bam = self.config.output_dir / f"{sam_file.stem}.sorted.bam"
            
            try:
                subprocess.run(["samtools", "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Warning: samtools not found. Skipping variant calling.")
                return None
            
            if not sam_file.exists():
                print(f"Warning: SAM file not found: {sam_file}. Skipping variant calling.")
                return None
            
            try:
                count_result = subprocess.run(
                    ["samtools", "view", "-c", "-F", "4", str(sam_file)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                
                if count_result.returncode != 0:
                    print(f"Warning: samtools validation failed: {count_result.stderr}")
                    with open(sam_file, "r") as f:
                        has_content = any(line.strip() and not line.startswith("@") for line in f)
                    if not has_content:
                        print("Warning: SAM file has no data lines. Skipping variant calling.")
                        return None
                else:
                    mapped_count = int(count_result.stdout.strip())
                    if mapped_count == 0:
                        print("Warning: No mapped reads in SAM file. Skipping variant calling.")
                        return None
            except subprocess.TimeoutExpired:
                print(f"Warning: SAM file validation timed out. Skipping variant calling.")
                return None
            except Exception as e:
                print(f"Warning: Failed to validate SAM file: {e}. Skipping variant calling.")
                return None
            
            try:
                with open(sam_file, "r") as sam_in, open(bam_file, "wb") as bam_out:
                    subprocess.run(
                        ["samtools", "view", "-bS", "-"],
                        stdin=sam_in,
                        stdout=bam_out,
                        stderr=subprocess.DEVNULL,
                        check=True,
                        timeout=120,
                    )
            except subprocess.TimeoutExpired:
                print(f"Warning: SAM to BAM conversion timed out after 120 seconds. Skipping variant calling.")
                if bam_file.exists():
                    try:
                        bam_file.unlink()
                    except Exception:
                        pass
                return None
            except FileNotFoundError:
                print(f"Warning: SAM file disappeared before conversion: {sam_file}")
                return None
            except subprocess.CalledProcessError as e:
                stderr_msg = e.stderr.decode('utf-8', errors='ignore').strip() if e.stderr else "No error details"
                print(f"Warning: SAM to BAM conversion failed: {stderr_msg}")
                if bam_file.exists():
                    try:
                        bam_file.unlink()
                    except Exception:
                        pass
                return None
            
            try:
                sort_result = subprocess.run(
                    ["samtools", "sort", "-o", str(sorted_bam), str(bam_file)],
                    check=True,
                    stderr=subprocess.PIPE,
                    timeout=180,
                )
            except subprocess.TimeoutExpired:
                print(f"Warning: BAM sorting timed out after 180 seconds. Skipping variant calling.")
                try:
                    if bam_file.exists():
                        bam_file.unlink()
                except Exception:
                    pass
                return None
            except subprocess.CalledProcessError as e:
                stderr_msg = e.stderr.decode('utf-8', errors='ignore').strip() if e.stderr else "No error details"
                print(f"Warning: BAM sorting failed: {stderr_msg}")
                try:
                    if bam_file.exists():
                        bam_file.unlink()
                except Exception:
                    pass
                return None
            
            if not sorted_bam.exists():
                print("Warning: Failed to create sorted BAM file. Skipping variant calling.")
                try:
                    if bam_file.exists():
                        bam_file.unlink()
                except Exception:
                    pass
                return None
            
            try:
                count_result = subprocess.run(
                    ["samtools", "view", "-c", "-F", "4", str(sorted_bam)],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=60,
                )
                mapped_count = int(count_result.stdout.strip())
                if mapped_count == 0:
                    print("Warning: No mapped reads in BAM file. Skipping variant calling.")
                    try:
                        bam_file.unlink()
                        sorted_bam.unlink()
                    except Exception:
                        pass
                    return None
            except subprocess.TimeoutExpired:
                print(f"Warning: BAM read count timed out. Skipping variant calling.")
                try:
                    bam_file.unlink()
                    sorted_bam.unlink()
                except Exception:
                    pass
                return None
            except (subprocess.CalledProcessError, ValueError) as e:
                print(f"Warning: Failed to count mapped reads: {e}. Skipping variant calling.")
                try:
                    bam_file.unlink()
                    sorted_bam.unlink()
                except Exception:
                    pass
                return None
            
            try:
                subprocess.run(
                    ["samtools", "index", str(sorted_bam)],
                    check=True,
                    stderr=subprocess.PIPE,
                    timeout=60,
                )
            except subprocess.TimeoutExpired:
                print(f"Warning: BAM indexing timed out. Continuing anyway (indexing is optional).")
            except subprocess.CalledProcessError as e:
                stderr_msg = e.stderr.decode('utf-8', errors='ignore').strip() if e.stderr else "No error details"
                print(f"Warning: BAM indexing failed: {stderr_msg}")
            
            try:
                subprocess.run(["bcftools", "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                if self._evaluation_mode:
                    print("  [Evaluation] ERROR: bcftools not found. Cannot compute truth metrics.")
                else:
                    print("Warning: bcftools not found. Skipping variant calling.")
                return None
            
            with open(output_vcf, "w") as vcf_out:
                mpileup = subprocess.Popen(
                    [
                        "bcftools", "mpileup",
                        "-f", str(reference_fasta),
                        "-Q", "0",
                        "-d", "10000",
                        "-C", "50",
                        str(sorted_bam),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                subprocess.run(
                    [
                        "bcftools", "call",
                        "-mv",
                        "--ploidy", "1",
                    ],
                    stdin=mpileup.stdout,
                    stdout=vcf_out,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                mpileup.wait()
            
            if not output_vcf.exists():
                if self._evaluation_mode:
                    print(f"  [Evaluation] ERROR: VCF file not created: {output_vcf}")
                return None
            
            vcf_has_variants = False
            try:
                with open(output_vcf, "r") as f:
                    for line in f:
                        if line.strip() and not line.startswith("#"):
                            vcf_has_variants = True
                            break
            except Exception as e:
                if self._evaluation_mode:
                    print(f"  [Evaluation] Warning: Failed to check VCF content: {e}")
            
            try:
                bam_file.unlink()
                sorted_bam.unlink()
                (sorted_bam.with_suffix(".bam.bai")).unlink()
            except Exception:
                pass
            
            return output_vcf
            
        except subprocess.CalledProcessError as e:
            print(f"Warning: Variant calling failed: {e}")
            if e.stderr:
                print(f"  Error details: {e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}")
            try:
                if bam_file.exists():
                    bam_file.unlink()
                if sorted_bam.exists():
                    sorted_bam.unlink()
                if (sorted_bam.with_suffix(".bam.bai")).exists():
                    (sorted_bam.with_suffix(".bam.bai")).unlink()
            except Exception:
                pass
            return None
        except Exception as e:
            print(f"Warning: Variant calling failed: {e}")
            try:
                if 'bam_file' in locals() and bam_file.exists():
                    bam_file.unlink()
                if 'sorted_bam' in locals() and sorted_bam.exists():
                    sorted_bam.unlink()
            except Exception:
                pass
            return None
    
    def _extract_chromosome_from_fasta(self, fasta_path: Path) -> str:
        """Extract chromosome name from FASTA file path or header."""
        if 'chr' in fasta_path.name.lower():
            import re
            match = re.search(r'chr([0-9XY]+)', fasta_path.name, re.IGNORECASE)
            if match:
                return f"chr{match.group(1)}"
        
        try:
            with open(fasta_path, 'r') as f:
                for line in f:
                    if line.startswith('>'):
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
        """Merge overlapping intervals into non-overlapping intervals."""
        if not intervals:
            return []
        
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = [sorted_intervals[0]]
        for current_start, current_end in sorted_intervals[1:]:
            last_start, last_end = merged[-1]
            if current_start <= last_end + 1:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
        
        return merged
    
    def _extract_coverage_regions_from_sam(self, sam_file: Path, current_chromosome: str = None) -> Optional[List[Tuple[int, int]]]:
        """Extract covered intervals from SAM file by collecting read intervals and merging overlaps."""
        if not sam_file or not sam_file.exists():
            return None
        
        intervals = []
        current_chromosome_norm = self._normalize_chromosome_name(current_chromosome) if current_chromosome else None
        
        try:
            with open(sam_file, "r") as f:
                for line in f:
                    if line.startswith("@"):
                        continue
                    
                    fields = line.strip().split("\t")
                    if len(fields) < 11:
                        continue
                    
                    try:
                        flag = int(fields[1])
                    except ValueError:
                        continue
                    if flag == 4:
                        continue
                    
                    chrom = fields[2]
                    pos_str = fields[3]
                    
                    if pos_str == '*' or pos_str == '0':
                        continue
                    
                    try:
                        pos = int(pos_str)
                    except ValueError:
                        continue
                    
                    if current_chromosome_norm:
                        chrom_norm = self._normalize_chromosome_name(chrom)
                        if chrom_norm != current_chromosome_norm:
                            continue
                    
                    cigar_str = fields[5]
                    if cigar_str == '*':
                        read_end = pos
                    else:
                        import re
                        cigar_ops = re.findall(r'(\d+)([MIDNSHPX=])', cigar_str)
                        reference_length = 0
                        for length_str, op in cigar_ops:
                            length = int(length_str)
                            if op in ['M', 'D', 'N', '=', 'X']:
                                reference_length += length
                        
                        read_end = pos + reference_length - 1
                    
                    intervals.append((pos, read_end))
                        
        except Exception as e:
            print(f"Warning: Failed to extract coverage regions from SAM: {e}")
            return None
        
        if not intervals:
            return None
        
        merged_intervals = self._merge_intervals(intervals)
        
        return merged_intervals
    
    def _compare_vcf_with_truth(self, query_vcf: Path, truth_vcf: Path, reference_fasta: Path, confident_regions: Path = None, current_chromosome: str = None, coverage_regions: Optional[List[Tuple[int, int]]] = None) -> Dict[str, float]:
        """Compare query VCF against truth VCF using Python-based comparison."""
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
            return self._compare_vcf_python(query_vcf, truth_vcf, current_chromosome=current_chromosome, coverage_regions=coverage_regions)
    
    def _compare_vcf_happy(self, query_vcf: Path, truth_vcf: Path, reference_fasta: Path, confident_regions: Path = None, current_chromosome: str = None) -> Dict[str, float]:
        """Compare VCFs using hap.py."""
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
                    if row.get("Type") == "SNP":
                        tp_snp = float(row.get("TP", 0))
                        fp_snp = float(row.get("FP", 0))
                        fn_snp = float(row.get("FN", 0))
                    elif row.get("Type") == "INDEL":
                        tp_indel = float(row.get("TP", 0))
                        fp_indel = float(row.get("FP", 0))
                        fn_indel = float(row.get("FN", 0))
            
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
        """Normalize chromosome name to 'chr1', 'chr2', etc. format."""
        chrom = str(chrom).strip()
        chrom_lower = chrom.lower()
        if chrom_lower.startswith('chr'):
            return chrom_lower
        else:
            return f"chr{chrom}"
    
    def _position_in_intervals(self, pos: int, intervals: List[Tuple[int, int]]) -> bool:
        """Check if a position is within any of the given intervals."""
        for start, end in intervals:
            if start <= pos <= end:
                return True
        return False
    
    def _compare_vcf_python(self, query_vcf: Path, truth_vcf: Path, current_chromosome: str = None, coverage_regions: Optional[List[Tuple[int, int]]] = None) -> Dict[str, float]:
        """Python-based VCF comparison."""
        if self._truth_vcf_cache is not None:
            all_truth_variants = self._truth_vcf_cache
        else:
            all_truth_variants = self._parse_vcf_variants(truth_vcf)
        
        current_chromosome_norm = None
        if current_chromosome:
            current_chromosome_norm = self._normalize_chromosome_name(current_chromosome)
            truth_variants = {
                key: var for key, var in all_truth_variants.items()
                if self._normalize_chromosome_name(var["chrom"]) == current_chromosome_norm
            }
        else:
            truth_variants = all_truth_variants
        
        if coverage_regions and len(coverage_regions) > 0:
            truth_variants = {
                key: var for key, var in truth_variants.items()
                if self._position_in_intervals(var["pos"], coverage_regions)
            }
        
        if len(truth_variants) == 0:
            return self._default_truth_metrics()
        
        query_variants_dict = self._parse_vcf_variants(query_vcf)
        query_variants_list = list(query_variants_dict.values())
        
        if current_chromosome:
            current_chromosome_norm = self._normalize_chromosome_name(current_chromosome)
            query_variants_list = [
                q_var for q_var in query_variants_list
                if self._normalize_chromosome_name(q_var["chrom"]) == current_chromosome_norm
            ]
        
        def normalize_key(key):
            chrom, pos, ref, alt = key
            chrom_norm = self._normalize_chromosome_name(chrom)
            return (chrom_norm, pos, ref, alt)
        
        query_keys_normalized = {normalize_key((q["chrom"], q["pos"], q["ref"], q["alt"])) 
                                 for q in query_variants_list}
        truth_keys_normalized = {normalize_key(key) for key in truth_variants.keys()}
        
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
        
        fn_snp = sum(1 for key, t_var in truth_variants.items() 
                    if t_var["type"] == "SNP" and normalize_key(key) not in query_keys_normalized)
        fn_indel = sum(1 for key, t_var in truth_variants.items() 
                      if t_var["type"] == "INDEL" and normalize_key(key) not in query_keys_normalized)
        
        tp_total = tp_snp + tp_indel
        fp_total = fp
        fn_total = fn_snp + fn_indel
        
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
        """Parse VCF file and return dictionary of variants."""
        import gzip
        
        variants = {}
        
        try:
            is_gzipped = False
            if vcf_file.suffix == ".gz":
                is_gzipped = True
            else:
                try:
                    with open(vcf_file, "rb") as test_f:
                        magic = test_f.read(2)
                        if magic == b'\x1f\x8b':
                            is_gzipped = True
                except Exception:
                    pass
            
            if is_gzipped:
                open_func = gzip.open
                mode = "rt"
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
        """Return default truth metrics."""
        return {
            "true_snps_tp": 0.0,
            "true_indels_tp": 0.0,
            "false_positives": 0.0,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
    
    def _parse_sam_metrics(self, sam_file: Path) -> Dict[str, float]:
        """Parse SAM file to extract mapping metrics and variant detection."""
        if not sam_file.exists():
            return self._default_metrics()
        
        try:
            with open(sam_file, "r") as f:
                lines = f.readlines()
            
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
                
                try:
                    flag = int(fields[1])
                except ValueError:
                    continue
                if flag != 4:
                    mapped_reads += 1
                    
                    cigar_str = fields[5]
                    
                    as_score = None
                    for field in fields[11:]:
                        if field.startswith("AS:i:"):
                            as_score = int(field.split(":")[2])
                            alignment_scores.append(as_score)
                            break
                    
                    nm_dist = None
                    md_tag = None
                    for field in fields[11:]:
                        if field.startswith("NM:i:"):
                            nm_dist = int(field.split(":")[2])
                            edit_distances.append(nm_dist)
                        elif field.startswith("MD:Z:"):
                            md_tag = field
                    
                    if cigar_str != '*':
                        variants = self._parse_cigar_variants(cigar_str)
                        indel_events = variants["indel_count"]
                        total_indels += indel_events
                        
                        if md_tag:
                            snps = self._parse_md_tag_for_snps(md_tag, cigar_str)
                            total_snps += snps
                        else:
                            if nm_dist is not None:
                                indel_bases = self._count_indel_bases(cigar_str)
                                estimated_snps = max(0, nm_dist - indel_bases)
                                total_snps += estimated_snps
            
            mapping_rate = mapped_reads / total_reads if total_reads > 0 else 0.0
            avg_alignment_score = np.mean(alignment_scores) if alignment_scores else 0.0
            avg_edit_distance = np.mean(edit_distances) if edit_distances else 0.0
            
            metrics = {
                "mapping_rate": mapping_rate,
                "alignment_score": avg_alignment_score,
                "edit_distance": avg_edit_distance,
                "indel_count": float(total_indels),
                "snp_count": float(total_snps),
                "total_variants": float(total_indels + total_snps),
            }
            
            return metrics
        except Exception as e:
            print(f"Warning: Failed to parse SAM file {sam_file}: {e}")
            return self._default_metrics()

    def _default_metrics(self) -> Dict[str, float]:
        """Return default metrics when parsing fails."""
        metrics = {
            "mapping_rate": 0.0,
            "alignment_score": 0.0,
            "edit_distance": self.config.edit_distance_hard_threshold,
            "indel_count": 0.0,
            "snp_count": 0.0,
            "total_variants": 0.0,
        }
        
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
    ) -> Tuple[float, Dict[str, float]]:
        """Compute reward from runtime, accuracy metrics, and variant detection."""
        if self._dataset_features:
            num_reads_normalized = self._dataset_features.get("num_reads", 0.0)
            
            if num_reads_normalized < self.config.small_dataset_threshold:
                runtime_weight = self.config.runtime_weight * self.config.small_dataset_runtime_weight_multiplier
                mapping_rate_weight = self.config.mapping_rate_weight * self.config.small_dataset_accuracy_weight_multiplier
                alignment_score_weight = self.config.alignment_score_weight * self.config.small_dataset_accuracy_weight_multiplier
                edit_distance_weight = self.config.edit_distance_weight * self.config.small_dataset_accuracy_weight_multiplier
            else:
                runtime_weight = self.config.runtime_weight
                mapping_rate_weight = self.config.mapping_rate_weight
                alignment_score_weight = self.config.alignment_score_weight
                edit_distance_weight = self.config.edit_distance_weight
        else:
            runtime_weight = self.config.runtime_weight
            mapping_rate_weight = self.config.mapping_rate_weight
            alignment_score_weight = self.config.alignment_score_weight
            edit_distance_weight = self.config.edit_distance_weight
        
        edit_distance = metrics["edit_distance"]
        
        if edit_distance <= self.config.edit_distance_soft_cap:
            edit_distance_penalty = edit_distance_weight * edit_distance
        else:
            edit_distance_penalty = edit_distance_weight * self.config.edit_distance_soft_cap
            excess_edit_distance = edit_distance - self.config.edit_distance_soft_cap
            extra_penalty = (
                self.config.edit_distance_high_penalty_multiplier 
                * abs(edit_distance_weight)
                * excess_edit_distance
            )
            edit_distance_penalty -= extra_penalty
        
        if edit_distance > self.config.edit_distance_critical_threshold:
            excess_critical = edit_distance - self.config.edit_distance_critical_threshold
            critical_penalty = (
                self.config.edit_distance_critical_penalty_multiplier
                * abs(edit_distance_weight)
                * excess_critical
            )
            edit_distance_penalty -= critical_penalty
        
        truth_metrics_available = False
        if self.config.use_truth_metrics and "true_snps_tp" in metrics:
            f1_check = metrics.get("f1_score", 0.0)
            true_indels_tp_check = metrics.get("true_indels_tp", 0.0)
            true_snps_tp_check = metrics.get("true_snps_tp", 0.0)
            truth_metrics_available = (f1_check > 0.0 or true_indels_tp_check > 0.0 or true_snps_tp_check > 0.0)
        
        effective_alignment_score_weight = alignment_score_weight
        if self.config.use_truth_metrics and not truth_metrics_available:
            effective_alignment_score_weight = 3.0
        
        runtime_normalized = runtime / self.config.runtime_normalization_factor
        
        runtime_contrib = runtime_weight * runtime_normalized
        mapping_rate_contrib = mapping_rate_weight * metrics["mapping_rate"]
        alignment_score_contrib = effective_alignment_score_weight * (metrics["alignment_score"] * 0.001)
        edit_distance_contrib = edit_distance_penalty
        
        reward = (
            runtime_contrib
            + mapping_rate_contrib
            + alignment_score_contrib
            + edit_distance_contrib
        )
        
        truth_contribs = {}
        
        if truth_metrics_available:
            f1_score = metrics.get("f1_score", 0.0)
            recall = metrics.get("recall", 0.0)
            precision = metrics.get("precision", 0.0)
            
            f1_scaled = f1_score * 1000.0
            recall_scaled = recall * 1000.0
            precision_scaled = precision * 1000.0
            
            true_snps_tp_contrib = self.config.true_snps_tp_weight * metrics.get("true_snps_tp", 0.0)
            true_indels_tp_contrib = self.config.true_indels_tp_weight * metrics.get("true_indels_tp", 0.0)
            f1_contrib = self.config.f1_score_weight * f1_scaled
            false_positives_contrib = self.config.false_positives_penalty * metrics.get("false_positives", 0.0)
            precision_contrib = 0.008 * precision_scaled
            recall_contrib = 0.018 * recall_scaled
            
            reward += (
                true_snps_tp_contrib
                + true_indels_tp_contrib
                + f1_contrib
                + false_positives_contrib
                + precision_contrib
                + recall_contrib
            )
            
            truth_contribs = {
                "true_snps_tp": true_snps_tp_contrib,
                "true_indels_tp": true_indels_tp_contrib,
                "f1": f1_contrib,
                "false_positives": false_positives_contrib,
                "precision": precision_contrib,
                "recall": recall_contrib,
            }
        else:
            snp_count_contrib = self.config.snp_count_weight * metrics.get("snp_count", 0.0)
            indel_count_contrib = self.config.indel_count_weight * metrics.get("indel_count", 0.0)
            total_variants_contrib = self.config.total_variants_weight * metrics.get("total_variants", 0.0)
            
            reward += (
                snp_count_contrib
                + indel_count_contrib
                + total_variants_contrib
            )
            
            truth_contribs = {
                "snp_count": snp_count_contrib,
                "indel_count": indel_count_contrib,
                "total_variants": total_variants_contrib,
            }
        
        normalization_factor = 1.0
        if self.config.normalize_weights:
            weight_magnitudes = [
                abs(runtime_weight),
                abs(mapping_rate_weight),
                abs(effective_alignment_score_weight),
                abs(edit_distance_weight),
            ]
            
            if truth_metrics_available:
                weight_magnitudes.extend([
                    abs(self.config.true_snps_tp_weight),
                    abs(self.config.true_indels_tp_weight),
                    abs(self.config.f1_score_weight),
                    abs(self.config.false_positives_penalty),
                    0.008,
                    0.018,
                ])
            else:
                weight_magnitudes.extend([
                    abs(self.config.snp_count_weight),
                    abs(self.config.indel_count_weight),
                    abs(self.config.total_variants_weight),
                ])
            
            mapping_rate_penalty_multiplier = 5.0
            weight_magnitudes.append(mapping_rate_penalty_multiplier)
            
            current_sum = sum(weight_magnitudes)
            if current_sum > 0:
                normalization_factor = 20.0 / current_sum
        
        reward = reward * normalization_factor
        
        mapping_rate_hard_penalty_contrib = 0.0
        edit_distance_hard_penalty_contrib = 0.0
        mapping_rate_soft_penalty_contrib = 0.0
        
        if metrics["mapping_rate"] < self.config.mapping_rate_hard_threshold:
            mapping_rate_hard_penalty_contrib = self.config.mapping_rate_hard_penalty
            reward += mapping_rate_hard_penalty_contrib
        
        if edit_distance > self.config.edit_distance_hard_threshold:
            edit_distance_hard_penalty_contrib = self.config.edit_distance_hard_penalty
            reward += edit_distance_hard_penalty_contrib
        
        if metrics["mapping_rate"] < self.config.min_mapping_rate_threshold:
            penalty_multiplier = 5.0 * normalization_factor
            mapping_rate_soft_penalty_contrib = -penalty_multiplier * (self.config.min_mapping_rate_threshold - metrics["mapping_rate"])
            reward += mapping_rate_soft_penalty_contrib
        
        contributions_raw = {
            "runtime": runtime_contrib,
            "mapping_rate": mapping_rate_contrib,
            "alignment_score": alignment_score_contrib,
            "edit_distance": edit_distance_contrib,
            **truth_contribs,
        }
        
        contributions_normalized = {
            key: val * normalization_factor 
            for key, val in contributions_raw.items()
        }
        
        contributions_normalized["mapping_rate_hard_penalty"] = mapping_rate_hard_penalty_contrib
        contributions_normalized["edit_distance_hard_penalty"] = edit_distance_hard_penalty_contrib
        contributions_normalized["mapping_rate_soft_penalty"] = mapping_rate_soft_penalty_contrib
        
        return float(reward), contributions_normalized

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
        reward, contributions = self._compute_reward(runtime, metrics, pattern_str=pattern_str)
        
        # Track contributions for analysis
        self._reward_contributions.append(contributions)
        
        # Log contribution statistics periodically
        if len(self._reward_contributions) >= self._log_contributions_freq:
            self._log_contribution_stats()
            self._reward_contributions = []  # Reset after logging
        
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
            "reward_contributions": contributions,  # Include in info for potential external logging
        }
        
        return self._current_observation, reward, terminated, truncated, info

    def _log_contribution_stats(self) -> None:
        """Log statistics about reward component contributions."""
        if not self._reward_contributions:
            return
        
        import numpy as np
        
        # Collect all component names
        all_keys = set()
        for contrib_dict in self._reward_contributions:
            all_keys.update(contrib_dict.keys())
        
        # Calculate statistics for each component
        stats = {}
        for key in sorted(all_keys):
            values = [contrib.get(key, 0.0) for contrib in self._reward_contributions]
            if values:
                stats[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "abs_mean": np.mean(np.abs(values)),
                }
        
        # Print formatted statistics
        print(f"\n{'='*80}")
        print(f"Reward Component Contributions (last {len(self._reward_contributions)} episodes)")
        print(f"{'='*80}")
        print(f"{'Component':<30} {'Mean':>12} {'Std':>12} {'Abs Mean':>12} {'Min':>12} {'Max':>12}")
        print(f"{'-'*80}")
        
        for key, stat in stats.items():
            print(
                f"{key:<30} "
                f"{stat['mean']:>12.4f} "
                f"{stat['std']:>12.4f} "
                f"{stat['abs_mean']:>12.4f} "
                f"{stat['min']:>12.4f} "
                f"{stat['max']:>12.4f}"
            )
        
        # Check for order-of-magnitude differences
        abs_means = {k: v['abs_mean'] for k, v in stats.items()}
        if abs_means:
            max_abs_mean = max(abs_means.values())
            print(f"\n{'='*80}")
            print("Order-of-magnitude check (components with abs_mean > 10× smallest):")
            min_abs_mean = min(v for v in abs_means.values() if v > 0)
            for key, abs_mean in sorted(abs_means.items(), key=lambda x: x[1], reverse=True):
                if abs_mean > 0 and abs_mean > 10 * min_abs_mean:
                    ratio = abs_mean / min_abs_mean if min_abs_mean > 0 else float('inf')
                    print(f"  {key:<30} abs_mean={abs_mean:>12.4f} ({ratio:>6.1f}× smallest)")
        print(f"{'='*80}\n")
    
    def set_evaluation_mode(self, evaluation_mode: bool = True) -> None:
        """Set evaluation mode flag.
        
        When True, variant calling will run every episode (not just every 5th).
        This ensures accurate truth metrics during evaluation.
        
        Args:
            evaluation_mode: If True, enable evaluation mode (variant calling every episode)
        """
        self._evaluation_mode = evaluation_mode
        if evaluation_mode:
            print("Evaluation mode enabled: variant calling will run every episode")
        else:
            print("Training mode: variant calling will run every 5th episode")

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
