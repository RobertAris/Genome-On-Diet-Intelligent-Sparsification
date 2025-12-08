"""Dataset manager for multi-chromosome training.

Manages multiple chromosome datasets and provides random selection for training.
"""
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ChromosomeDatasetManager:
    """Manages multiple chromosome datasets for RL training."""
    
    def __init__(
        self,
        chromosomes_dir: Path,
        query_fastq: Path = None,
        query_fastq_dir: Path = None,
        chromosomes: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize dataset manager.
        
        Args:
            chromosomes_dir: Directory containing chromosome FASTA files
            query_fastq: Path to query FASTQ file (shared across chromosomes) - DEPRECATED
            query_fastq_dir: Directory containing chromosome-specific FASTQ files (chr1_reads.fastq, etc.)
            chromosomes: List of chromosome names to use (e.g., ['chr1', 'chr2'])
                        If None, auto-detects available chromosomes
            seed: Random seed for reproducibility
        """
        self.chromosomes_dir = Path(chromosomes_dir)
        self.query_fastq_dir = Path(query_fastq_dir) if query_fastq_dir else None
        self.query_fastq = Path(query_fastq) if query_fastq else None  # Legacy support
        
        if seed is not None:
            random.seed(seed)
        
        # Auto-detect chromosomes if not specified
        if chromosomes is None:
            chromosomes = self._detect_chromosomes()
        
        self.chromosomes = sorted(chromosomes)
        self._validate_chromosomes()
        
        print(f"Dataset Manager initialized with {len(self.chromosomes)} chromosomes:")
        for chr_name in self.chromosomes:
            print(f"  - {chr_name}")
    
    def _detect_chromosomes(self) -> List[str]:
        """Auto-detect available chromosome files."""
        chromosomes = []
        for fasta_file in sorted(self.chromosomes_dir.glob("chr*.fasta")):
            chr_name = fasta_file.stem
            chromosomes.append(chr_name)
        return chromosomes
    
    def _validate_chromosomes(self):
        """Validate that all chromosome files exist."""
        missing = []
        for chr_name in self.chromosomes:
            chr_file = self.chromosomes_dir / f"{chr_name}.fasta"
            if not chr_file.exists():
                missing.append(chr_name)
        
        if missing:
            raise FileNotFoundError(
                f"Missing chromosome files: {missing}\n"
                f"Expected in: {self.chromosomes_dir}"
            )
        
        # Validate query FASTQ files
        if self.query_fastq_dir:
            # Chromosome-specific FASTQ files mode
            for chr_name in self.chromosomes:
                fastq_file = self.query_fastq_dir / f"{chr_name}_reads.fastq"
                if not fastq_file.exists():
                    raise FileNotFoundError(
                        f"Chromosome-specific FASTQ not found: {fastq_file}\n"
                        f"Run: python3 rl/scripts/extract_chromosome_reads.py --sam-file <sam> --input-fastq <fastq> --output-dir {self.query_fastq_dir}"
                    )
        elif self.query_fastq:
            # Legacy: shared FASTQ file
            if not self.query_fastq.exists():
                raise FileNotFoundError(f"Query FASTQ not found: {self.query_fastq}")
        else:
            raise ValueError("Either query_fastq_dir or query_fastq must be provided")
    
    def get_random_dataset(self) -> Tuple[Path, Path, str]:
        """
        Get a random chromosome dataset.
        
        Returns:
            (reference_fasta, query_fastq, chromosome_name)
        """
        chromosome = random.choice(self.chromosomes)
        reference_fasta = self.chromosomes_dir / f"{chromosome}.fasta"
        
        # Use chromosome-specific FASTQ if available
        if self.query_fastq_dir:
            query_fastq = self.query_fastq_dir / f"{chromosome}_reads.fastq"
        else:
            query_fastq = self.query_fastq  # Legacy: shared FASTQ
        
        return reference_fasta, query_fastq, chromosome
    
    def get_dataset(self, chromosome: str) -> Tuple[Path, Path, str]:
        """
        Get a specific chromosome dataset.
        
        Args:
            chromosome: Chromosome name (e.g., 'chr1')
        
        Returns:
            (reference_fasta, query_fastq, chromosome_name)
        """
        if chromosome not in self.chromosomes:
            raise ValueError(
                f"Chromosome '{chromosome}' not available. "
                f"Available: {self.chromosomes}"
            )
        
        reference_fasta = self.chromosomes_dir / f"{chromosome}.fasta"
        
        # Use chromosome-specific FASTQ if available
        if self.query_fastq_dir:
            query_fastq = self.query_fastq_dir / f"{chromosome}_reads.fastq"
        else:
            query_fastq = self.query_fastq  # Legacy: shared FASTQ
        
        return reference_fasta, query_fastq, chromosome
    
    def get_all_datasets(self) -> List[Tuple[Path, Path, str]]:
        """
        Get all available datasets.
        
        Returns:
            List of (reference_fasta, query_fastq, chromosome_name) tuples
        """
        return [self.get_dataset(chr_name) for chr_name in self.chromosomes]
    
    @property
    def num_datasets(self) -> int:
        """Number of available datasets."""
        return len(self.chromosomes)

