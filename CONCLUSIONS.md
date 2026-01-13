# Project Conclusions: RL-Based Sparsification Pattern Selection for Genome-on-Diet

## Executive Summary

This project demonstrates that reinforcement learning can learn sparsification patterns for Illumina short read mapping using Genome-on-Diet. The trained PPO agent outperforms a random baseline, achieving 30% higher reward, 19% faster runtime, 1.76% higher mapping rate, and 28% better edit distance.

---

## Key Findings

### Performance Improvements

| Metric | Trained Model | Random Baseline | Improvement |
|--------|---------------|-----------------|-------------|
| Average Reward | 43.83 | 33.78 | +30.0% |
| Runtime | 2.14s | 2.64s | -19.0% |
| Mapping Rate | 92.45% | 90.69% | +1.76% |
| Alignment Score | 439.56 | 418.81 | +5.0% |
| Edit Distance | 8.34 | 11.63 | -28.3% |

### Pattern Selection

The model adapts pattern selection based on dataset characteristics:
- **Pattern '01' (50% sparsification)**: Selected 65% of the time, average runtime ~2.8s, mapping rate ~90.9%
- **Pattern '111' (0% sparsification)**: Selected 35% of the time, average runtime ~0.8s, mapping rate ~95.3%

The model learns to use less sparsification for smaller datasets. The baseline always selected pattern '10' (50% sparsification) with no adaptation.

### Variant Detection

| Metric | Trained Model | Random Baseline |
|--------|---------------|-----------------|
| Precision | 54.0% | 61.0% |
| Recall | 32.6% | 25.4% |
| F1 Score | 0.3866 | 0.3574 |

The trained model achieves 8% higher F1 score and 28% higher recall, though precision is lower. Both models show 0 true positive indels.

---

## Technical Insights

The model learns to use less sparsification for smaller datasets, explaining why pattern '111' achieves faster runtime (0.8s) than pattern '01' (2.8s) in those cases. Training across 8 chromosomes (chr1, chr2, chr3, chr4, chr5, chr21, chr22, chrX) enables adaptation to different reference sizes and dataset characteristics.

---

## Limitations and Future Work

### Current Limitations

1. **Training Duration**: The model requires longer training to reach full potential and improve stability.

2. **Training Data**: Model trained on a subset of chromosomes. Full genome training could improve generalization.

3. **Variant Calling**: Indel detection shows 0 true positives for both models, suggesting room for improvement.

### Future Work

1. **Multi-Objective Optimization**: Optimize for precision-recall curves or AUC.

2. **Transfer Learning**: Pre-train on synthetic data or other genomes, then fine-tune on target datasets.

3. **Online Learning**: Adapt patterns during runtime based on intermediate mapping results.

4. **Ensemble Methods**: Combine multiple pattern selections for robust performance.

---

## Conclusions

The PPO agent learns context-aware sparsification patterns that outperform fixed baselines, achieving 30% higher reward, 19% faster runtime, and 28% better edit distance. The model's adaptive selection (65% '01', 35% '111') based on dataset characteristics learns to use less sparsification for smaller datasets. Multi-chromosome training enables generalization across different reference sizes. The model requires longer training to reach full potential.
