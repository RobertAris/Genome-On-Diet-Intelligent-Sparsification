# Truth-Based Metrics Data

This directory contains benchmark data for truth-based variant evaluation.

## Files

- `HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz` - HG002 truth VCF file (149 MB)
  - Downloaded from: https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG002_NA24385_son/NISTv4.2.1/GRCh38/
  - Contains validated variants for chromosomes 1-22

## Optional Files

- `HG002_GRCh38_v4.2.1_benchmark.bed` - Confident regions BED file (optional)
  - Can be downloaded manually if needed for hap.py evaluation
  - Restricts evaluation to high-confidence regions only

## Tools Installed

- **samtools** (v1.22.1) - For SAM/BAM file manipulation
- **bcftools** (v1.22) - For variant calling from BAM files
- **hap.py** - VCF comparison tool (Python fallback available if hap.py installation fails)

## Usage

To enable truth-based metrics in your RL training:

1. Set `use_truth_metrics: true` in your config file
2. Ensure `truth_vcf` path points to the VCF file above
3. The system will automatically:
   - Call variants from SAM files using bcftools
   - Compare against the truth VCF
   - Calculate TP, FP, F1, precision, and recall

## Notes

- The VCF file is gzipped and will be automatically decompressed when used
- If hap.py is not available, the system falls back to a Python-based VCF comparison
- The BED file is optional - evaluation works without it, but may include low-confidence regions

