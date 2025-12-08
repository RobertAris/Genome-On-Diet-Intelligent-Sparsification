"""Extract reads by chromosome from a full genome FASTQ using a SAM file.

This script uses a pre-computed SAM file (from mapping full FASTQ to full genome)
to extract reads that map to each chromosome into separate FASTQ files.
"""
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set


def parse_sam_for_chromosomes(sam_file: Path) -> Dict[str, Set[str]]:
    """
    Parse SAM file to find which reads map to which chromosomes.
    
    Returns:
        Dictionary mapping chromosome name to set of read names
    """
    chromosome_reads = defaultdict(set)
    
    with open(sam_file, 'r') as f:
        for line in f:
            if line.startswith('@'):
                continue  # Skip header lines
            
            fields = line.strip().split('\t')
            if len(fields) < 11:
                continue
            
            # Check if mapped (flag != 4)
            flag = int(fields[1])
            if flag == 4:  # Unmapped
                continue
            
            # Field 0: QNAME (read name)
            # Field 2: RNAME (reference name/chromosome)
            read_name = fields[0]
            chromosome = fields[2]
            
            if chromosome != '*':  # Mapped to a chromosome
                chromosome_reads[chromosome].add(read_name)
    
    return chromosome_reads


def extract_reads_from_fastq(
    input_fastq: Path,
    output_fastq: Path,
    read_names: Set[str]
) -> int:
    """
    Extract reads from FASTQ file that are in the read_names set.
    
    Returns:
        Number of reads extracted
    """
    count = 0
    current_read = None
    read_buffer = []
    
    with open(input_fastq, 'r') as infile, open(output_fastq, 'w') as outfile:
        for line in infile:
            if line.startswith('@'):
                # New read header
                if current_read and current_read in read_names:
                    # Write previous read if it was in our set
                    outfile.writelines(read_buffer)
                    count += 1
                
                # Start new read
                read_name = line.strip().split()[0][1:]  # Remove @ and get read name
                current_read = read_name
                read_buffer = [line]
            else:
                # Continue current read (sequence, +, quality)
                read_buffer.append(line)
        
        # Don't forget the last read
        if current_read and current_read in read_names:
            outfile.writelines(read_buffer)
            count += 1
    
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Extract chromosome-specific reads from FASTQ using SAM file"
    )
    parser.add_argument(
        '--sam-file',
        type=Path,
        required=True,
        help='SAM file from mapping full FASTQ to full genome'
    )
    parser.add_argument(
        '--input-fastq',
        type=Path,
        required=True,
        help='Input FASTQ file (full genome reads)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for chromosome-specific FASTQ files'
    )
    parser.add_argument(
        '--chromosomes',
        nargs='+',
        help='Specific chromosomes to extract (default: all found in SAM)'
    )
    parser.add_argument(
        '--max-reads-per-chromosome',
        type=int,
        default=None,
        help='Maximum number of reads to extract per chromosome (default: all available). '
             'Useful for creating dataset size diversity (e.g., chr1: 1000, chr21: 100)'
    )
    parser.add_argument(
        '--reads-per-chromosome',
        type=str,
        default=None,
        help='Comma-separated list of chromosome:reads pairs (e.g., "chr1:1000,chr2:800,chr21:100"). '
             'Overrides --max-reads-per-chromosome for specific chromosomes'
    )
    parser.add_argument(
        '--use-diversity-defaults',
        action='store_true',
        help='Use default read counts to create dataset size diversity: '
             'Large chromosomes (chr1, chr2): 1000 reads, '
             'Medium (chr3, chr4, chr5): 600 reads, '
             'Small (chr21, chr22): 150 reads, '
             'chrX: 300 reads'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Parsing SAM file: {args.sam_file}")
    chromosome_reads = parse_sam_for_chromosomes(args.sam_file)
    
    # Filter chromosomes if specified
    if args.chromosomes:
        chromosome_reads = {
            chr: reads for chr, reads in chromosome_reads.items()
            if chr in args.chromosomes
        }
    
    print(f"Found {len(chromosome_reads)} chromosomes in SAM file")
    
    # Default diversity settings: larger chromosomes get more reads
    diversity_defaults = {
        'chr1': 1000,
        'chr2': 1000,
        'chr3': 600,
        'chr4': 600,
        'chr5': 600,
        'chr21': 150,
        'chr22': 150,
        'chrX': 300,
    }
    
    # Parse chromosome-specific read limits if provided
    chromosome_limits = {}
    if args.use_diversity_defaults:
        # Use diversity defaults for chromosomes that exist
        for chr_name in chromosome_reads.keys():
            if chr_name in diversity_defaults:
                chromosome_limits[chr_name] = diversity_defaults[chr_name]
        print(f"Using diversity defaults: {chromosome_limits}")
    elif args.reads_per_chromosome:
        for pair in args.reads_per_chromosome.split(','):
            if ':' in pair:
                chr_name, limit = pair.split(':')
                chromosome_limits[chr_name.strip()] = int(limit.strip())
    
    # Extract reads for each chromosome
    total_extracted = 0
    for chromosome, read_names in sorted(chromosome_reads.items()):
        output_file = args.output_dir / f"{chromosome}_reads.fastq"
        
        # Determine how many reads to extract
        available_reads = len(read_names)
        if chromosome in chromosome_limits:
            # Use chromosome-specific limit
            max_reads = chromosome_limits[chromosome]
        elif args.max_reads_per_chromosome:
            # Use global limit
            max_reads = args.max_reads_per_chromosome
        else:
            # Extract all available reads
            max_reads = available_reads
        
        # Limit the read set if needed
        if max_reads < available_reads:
            # Convert to list, shuffle for randomness, then limit
            import random
            read_list = list(read_names)
            random.seed(42)  # For reproducibility
            random.shuffle(read_list)
            read_names_to_extract = set(read_list[:max_reads])
            print(f"Extracting {max_reads} of {available_reads} available reads for {chromosome}...")
        else:
            read_names_to_extract = read_names
            print(f"Extracting all {available_reads} reads for {chromosome}...")
        
        count = extract_reads_from_fastq(
            args.input_fastq,
            output_file,
            read_names_to_extract
        )
        
        print(f"  → Extracted {count} reads to {output_file}")
        total_extracted += count
    
    print(f"\nTotal reads extracted: {total_extracted}")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()

