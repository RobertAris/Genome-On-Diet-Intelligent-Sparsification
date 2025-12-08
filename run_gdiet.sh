#!/bin/bash

###############################################################################
# Genome-on-Diet Mapping Script
# 
# This script runs GDiet with configurable sparsification pattern parameters
# and generates readable output summaries.
#
# Sparsification Pattern Parameters:
#   -Z STR: Diet pattern (string of '1's and '0's)
#          '1' = keep this position, '0' = skip this position
#          Examples:
#            "10" (W=2) = keep every other base (50% sparsification)
#            "110" (W=3) = keep 2 out of 3 bases (67% sparsification)
#            "111000111" (W=9) = keep first 3, skip 3, keep 3 (67% sparsification)
#   -W INT: Length of the diet pattern
#
# The pattern repeats across the sequence, so "10" with W=2 means:
#   Position: 0 1 2 3 4 5 6 7 8 9 ...
#   Pattern:  1 0 1 0 1 0 1 0 1 0 ...
#   Result:   Keep Skip Keep Skip Keep Skip Keep Skip Keep Skip ...
###############################################################################

# Default parameters (from paper: README.md line 44)
# Illumina sequences example: -Z 10 -W 2 -i 2 -k 21 -w 11 -N 1 -r 0.05,150,200 -n 0.95,0.3 -s 100 --AF_max_loc 2 --secondary=yes
DEFAULT_Z="10"           # Diet pattern: "10" = 50% sparsification (keeps every other base)
DEFAULT_W=2              # Pattern length: 2 bases
DEFAULT_K=21             # k-mer size
DEFAULT_WINDOW=11        # Minimizer window size
DEFAULT_THREADS=1        # Number of threads
DEFAULT_READ_TYPE="sr"   # Read type: sr (short reads), map-ont (long reads), map-hifi
DEFAULT_MAX_SEEDS=2      # Maximum number of minimizers for pattern alignment (-i)
DEFAULT_SECONDARY=1      # Number of secondary alignments (-N)
DEFAULT_BW="0.05,150,200" # Edit distance threshold (-r)
DEFAULT_MIN_CNT="0.95,0.3" # Retain mapping locations (-n)
DEFAULT_MIN_SCORE=100    # Minimal peak DP alignment score (-s)
DEFAULT_AF_MAX_LOC=2     # Retain at most INT mapping locations (--AF_max_loc)

# Parse command line arguments
Z_PATTERN="${Z_PATTERN:-$DEFAULT_Z}"
W_LENGTH="${W_LENGTH:-$DEFAULT_W}"
K_SIZE="${K_SIZE:-$DEFAULT_K}"
WINDOW_SIZE="${WINDOW_SIZE:-$DEFAULT_WINDOW}"
THREADS="${THREADS:-$DEFAULT_THREADS}"
READ_TYPE="${READ_TYPE:-$DEFAULT_READ_TYPE}"
MAX_SEEDS="${MAX_SEEDS:-$DEFAULT_MAX_SEEDS}"
SECONDARY="${SECONDARY:-$DEFAULT_SECONDARY}"
BW="${BW:-$DEFAULT_BW}"
MIN_CNT="${MIN_CNT:-$DEFAULT_MIN_CNT}"
MIN_SCORE="${MIN_SCORE:-$DEFAULT_MIN_SCORE}"
AF_MAX_LOC="${AF_MAX_LOC:-$DEFAULT_AF_MAX_LOC}"

# Input files (modify these paths as needed)
REFERENCE="${REFERENCE:-Data/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta}"
QUERY="${QUERY:-Data/D1_S1_L001_R1_001-017.fastq}"

# Output file
OUTPUT_PREFIX="${OUTPUT_PREFIX:-Genome-on-Diet-GRCh38-Illumina}"
OUTPUT_SAM="${OUTPUT_PREFIX}_k${K_SIZE}w${WINDOW_SIZE}_Z${Z_PATTERN}W${W_LENGTH}.sam"
OUTPUT_SUMMARY="${OUTPUT_PREFIX}_k${K_SIZE}w${WINDOW_SIZE}_Z${Z_PATTERN}W${W_LENGTH}_summary.txt"

# GDiet executable (use ShortReads or LongReads based on read type)
if [[ "$READ_TYPE" == "sr" ]]; then
    GDIET_EXE="GDiet-ShortReads/GDiet"
else
    GDIET_EXE="GDiet-ShortReads/GDiet"  # Updated for Illumina short reads only
fi

# Check if GDiet executable exists
if [[ ! -f "$GDIET_EXE" ]]; then
    echo "Error: GDiet executable not found at $GDIET_EXE"
    echo "Please build GDiet first or update GDIET_EXE path"
    exit 1
fi

# Check if input files exist
if [[ ! -f "$REFERENCE" ]]; then
    echo "Error: Reference file not found: $REFERENCE"
    exit 1
fi

if [[ ! -f "$QUERY" ]]; then
    echo "Error: Query file not found: $QUERY"
    exit 1
fi

# Calculate sparsification percentage
ONES_COUNT=$(echo "$Z_PATTERN" | grep -o "1" | wc -l | tr -d ' ')
TOTAL_LENGTH=${#Z_PATTERN}
SPARSIFICATION_PCT=$(awk "BEGIN {printf \"%.1f\", ($ONES_COUNT / $TOTAL_LENGTH) * 100}")

echo "=========================================="
echo "Genome-on-Diet Mapping"
echo "=========================================="
echo "Reference: $REFERENCE"
echo "Query:     $QUERY"
echo "Output:    $OUTPUT_SAM"
echo ""
echo "Sparsification Pattern:"
echo "  Pattern (-Z): '$Z_PATTERN'"
echo "  Length (-W):  $W_LENGTH"
echo "  Keep positions: $ONES_COUNT out of $TOTAL_LENGTH"
echo "  Sparsification: ${SPARSIFICATION_PCT}% of bases kept"
echo ""
echo "Other Parameters:"
echo "  k-mer size (-k):     $K_SIZE"
echo "  Window size (-w):    $WINDOW_SIZE"
echo "  Max seeds (-i):      $MAX_SEEDS"
echo "  Secondary (-N):      $SECONDARY"
echo "  Bandwidth (-r):      $BW"
echo "  Min count (-n):      $MIN_CNT"
echo "  Min score (-s):      $MIN_SCORE"
echo "  AF max loc:          $AF_MAX_LOC"
echo "  Threads (-t):        $THREADS"
echo "  Read type (-x):      $READ_TYPE"
echo "=========================================="
echo ""

# Run GDiet with parameters from the paper
echo "Running GDiet..."
$GDIET_EXE \
    -t $THREADS \
    -x $READ_TYPE \
    -Z $Z_PATTERN \
    -W $W_LENGTH \
    -k $K_SIZE \
    -w $WINDOW_SIZE \
    -i $MAX_SEEDS \
    -N $SECONDARY \
    -r $BW \
    -n $MIN_CNT \
    -s $MIN_SCORE \
    --AF_max_loc $AF_MAX_LOC \
    --secondary=yes \
    -a \
    -o "$OUTPUT_SAM" \
    "$REFERENCE" \
    "$QUERY"

EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "Error: GDiet failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "Generating summary statistics..."

# Generate summary statistics
{
    echo "=========================================="
    echo "Genome-on-Diet Mapping Summary"
    echo "=========================================="
    echo "Date: $(date)"
    echo ""
    echo "Parameters:"
    echo "  Reference: $REFERENCE"
    echo "  Query: $QUERY"
    echo "  Sparsification Pattern: '$Z_PATTERN' (length: $W_LENGTH, ${SPARSIFICATION_PCT}% kept)"
    echo "  k-mer size: $K_SIZE"
    echo "  Window size: $WINDOW_SIZE"
    echo "  Threads: $THREADS"
    echo ""
    echo "Output File: $OUTPUT_SAM"
    echo ""
    echo "----------------------------------------"
    echo "Mapping Statistics:"
    echo "----------------------------------------"
    
    # Count total reads
    TOTAL_READS=$(grep -v "^@" "$OUTPUT_SAM" | wc -l | tr -d ' ')
    echo "Total reads processed: $TOTAL_READS"
    
    # Count mapped reads (not unmapped, flag != 4)
    MAPPED_READS=$(grep -v "^@" "$OUTPUT_SAM" | awk '$2 != 4' | wc -l | tr -d ' ')
    UNMAPPED_READS=$((TOTAL_READS - MAPPED_READS))
    MAPPING_RATE=$(awk "BEGIN {printf \"%.2f\", ($MAPPED_READS / $TOTAL_READS) * 100}")
    echo "Mapped reads: $MAPPED_READS ($MAPPING_RATE%)"
    echo "Unmapped reads: $UNMAPPED_READS"
    
    # Count primary alignments
    PRIMARY_ALIGNMENTS=$(grep -v "^@" "$OUTPUT_SAM" | awk '$2 != 4 && ($2 < 256 || $2 >= 2048)' | wc -l | tr -d ' ')
    echo "Primary alignments: $PRIMARY_ALIGNMENTS"
    
    # Count secondary alignments
    SECONDARY_ALIGNMENTS=$(grep -v "^@" "$OUTPUT_SAM" | awk '$2 >= 256 && $2 < 2048' | wc -l | tr -d ' ')
    echo "Secondary alignments: $SECONDARY_ALIGNMENTS"
    
    # Calculate average mapping quality (if available)
    if grep -q "MQ:" "$OUTPUT_SAM"; then
        AVG_MAPQ=$(grep -v "^@" "$OUTPUT_SAM" | awk '$2 != 4 {sum+=$5; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
        echo "Average mapping quality: $AVG_MAPQ"
    fi
    
    # Calculate average alignment score (from AS tag)
    AVG_AS=$(grep -v "^@" "$OUTPUT_SAM" | grep "AS:i:" | sed 's/.*AS:i:\([0-9]*\).*/\1/' | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    if [[ "$AVG_AS" != "N/A" ]]; then
        echo "Average alignment score: $AVG_AS"
    fi
    
    # Calculate average edit distance (from NM tag)
    AVG_NM=$(grep -v "^@" "$OUTPUT_SAM" | grep "NM:i:" | sed 's/.*NM:i:\([0-9]*\).*/\1/' | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    if [[ "$AVG_NM" != "N/A" ]]; then
        echo "Average edit distance (NM): $AVG_NM"
    fi
    
    # Count reads by chromosome
    echo ""
    echo "----------------------------------------"
    echo "Mappings by Chromosome:"
    echo "----------------------------------------"
grep -v "^@" "$OUTPUT_SAM" | awk '$2 != 4 && $3 != "*" {print $3}' | sort | uniq -c | sort -rn | \
        awk '{printf "  %-20s %8d reads\n", $2, $1}'
    
    echo ""
    echo "=========================================="
    echo "Top 10 Alignments (by alignment score):"
    echo "=========================================="
    grep -v "^@" "$OUTPUT_SAM" | grep "AS:i:" | \
        sed 's/\([^\t]*\)\t\([^\t]*\)\t\([^\t]*\)\t\([^\t]*\)\t\([^\t]*\).*AS:i:\([0-9]*\).*/AS:\6\tRead:\1\tChr:\3\tPos:\4/' | \
        sort -t: -k2 -rn | head -10 | \
        awk -F'\t' '{printf "  %-10s %-30s %-15s %s\n", $1, $2, $3, $4}'
    
    echo ""
    echo "Summary saved to: $OUTPUT_SUMMARY"
    echo "=========================================="
} > "$OUTPUT_SUMMARY"

# Display summary
cat "$OUTPUT_SUMMARY"

echo ""
echo "Done! Output files:"
echo "  SAM file:    $OUTPUT_SAM"
echo "  Summary:     $OUTPUT_SUMMARY"

