#!/bin/bash
# Verification script for truth-based metrics setup

echo "=========================================="
echo "Truth-Based Metrics Setup Verification"
echo "=========================================="
echo ""

# Check samtools
echo -n "Checking samtools... "
if command -v samtools &> /dev/null; then
    VERSION=$(samtools --version | head -1)
    echo "✓ Found: $VERSION"
else
    echo "✗ NOT FOUND - Install with: brew install samtools"
    exit 1
fi

# Check bcftools
echo -n "Checking bcftools... "
if command -v bcftools &> /dev/null; then
    VERSION=$(bcftools --version | head -1)
    echo "✓ Found: $VERSION"
else
    echo "✗ NOT FOUND - Install with: brew install bcftools"
    exit 1
fi

# Check truth VCF file
echo -n "Checking truth VCF file... "
VCF_FILE="Data/truth_data/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
if [ -f "$VCF_FILE" ]; then
    SIZE=$(du -h "$VCF_FILE" | cut -f1)
    if gunzip -t "$VCF_FILE" 2>/dev/null; then
        echo "✓ Found and valid ($SIZE)"
    else
        echo "✗ Found but corrupted"
        exit 1
    fi
else
    echo "✗ NOT FOUND at $VCF_FILE"
    exit 1
fi

# Check config file
echo -n "Checking config file... "
CONFIG_FILE="rl/configs/env_multi_chromosome.yaml"
if [ -f "$CONFIG_FILE" ]; then
    if grep -q "use_truth_metrics: true" "$CONFIG_FILE"; then
        echo "✓ Found and truth metrics enabled"
    else
        echo "⚠ Found but truth metrics not enabled"
        echo "  Set 'use_truth_metrics: true' in $CONFIG_FILE"
    fi
else
    echo "✗ NOT FOUND at $CONFIG_FILE"
    exit 1
fi

# Check hap.py (optional)
echo -n "Checking hap.py (optional)... "
if command -v hap.py &> /dev/null; then
    echo "✓ Found (will use hap.py for VCF comparison)"
elif python3 -c "import happy" 2>/dev/null; then
    echo "⚠ Python 'happy' package found (different from hap.py)"
    echo "  Will use Python fallback for VCF comparison"
else
    echo "⚠ Not found (will use Python fallback for VCF comparison)"
fi

echo ""
echo "=========================================="
echo "Setup Status: ✓ Ready for truth-based metrics"
echo "=========================================="

