"""Split the full genome FASTA into individual chromosome files.

This script extracts each chromosome from the full genome FASTA file
and saves them as separate files for multi-chromosome training.
"""
import argparse
from pathlib import Path
from typing import List


def split_genome(input_fasta: Path, output_dir: Path, chromosomes: List[str] = None):
    """
    Split genome FASTA into individual chromosome files.
    
    Args:
        input_fasta: Path to full genome FASTA file
        output_dir: Directory to save chromosome files
        chromosomes: List of chromosomes to extract (e.g., ['chr1', 'chr2', ...])
                    If None, extracts all chromosomes found in the file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if chromosomes is None:
        # Common human chromosomes (1-22, X, Y, M)
        chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
    
    print(f"Splitting {input_fasta} into chromosome files...")
    print(f"Output directory: {output_dir}")
    print(f"Chromosomes to extract: {chromosomes}")
    print()
    
    current_chr = None
    current_file = None
    extracted_chromosomes = []
    
    with open(input_fasta, "r") as f:
        for line in f:
            if line.startswith(">"):
                # Parse chromosome name from header
                # Format: >chr1 AC:CM000663.2 ...
                header_parts = line[1:].strip().split()
                if header_parts:
                    chr_name = header_parts[0]
                    
                    # Check if this is a chromosome we want
                    if chr_name in chromosomes:
                        # Close previous file if open
                        if current_file:
                            current_file.close()
                        
                        # Open new chromosome file
                        output_file = output_dir / f"{chr_name}.fasta"
                        current_file = open(output_file, "w")
                        current_chr = chr_name
                        extracted_chromosomes.append(chr_name)
                        print(f"Extracting {chr_name}...")
                        
                        # Write header
                        current_file.write(line)
                    else:
                        # Not a chromosome we want, skip
                        current_file = None
                        current_chr = None
            else:
                # Write sequence line if we're in a chromosome we want
                if current_file:
                    current_file.write(line)
    
    # Close last file
    if current_file:
        current_file.close()
    
    print()
    print(f"✓ Extracted {len(extracted_chromosomes)} chromosomes:")
    for chr_name in extracted_chromosomes:
        file_path = output_dir / f"{chr_name}.fasta"
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  {chr_name}: {size_mb:.1f} MB")
    
    return extracted_chromosomes


def main():
    parser = argparse.ArgumentParser(
        description="Split full genome FASTA into individual chromosome files"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Data/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta"),
        help="Input full genome FASTA file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/chromosomes"),
        help="Output directory for chromosome files",
    )
    parser.add_argument(
        "--chromosomes",
        nargs="+",
        default=None,
        help="Specific chromosomes to extract (e.g., chr1 chr2 chr21). If not specified, extracts all standard chromosomes.",
    )
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    split_genome(args.input, args.output_dir, args.chromosomes)
    print(f"\n✓ Chromosome files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

