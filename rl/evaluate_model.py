"""Evaluate a trained PPO model on the GenomeDietPatternEnv."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:
    raise ImportError(
        "stable-baselines3 is required. Install via `pip install stable-baselines3`"
    ) from exc

# Add rl directory to path
sys.path.append(str(Path(__file__).resolve().parent))
from envs.genome_diet_pattern_env import load_env_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained PPO model")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).parent / "models" / "ppo_genome_diet.zip",
        help="Path to trained model (.zip file)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "env_multi_chromosome.yaml",
        help="Path to environment config YAML",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare against baseline pattern '10' (50% sparsification)",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def evaluate_model(
    model: PPO,
    env,
    n_episodes: int,
) -> Dict[str, Any]:
    """Evaluate the model and return statistics."""
    results = {
        "episodes": [],
        "total_reward": 0.0,
        "pattern_counts": Counter(),  # Count unique patterns
        "avg_runtime": 0.0,
        "avg_mapping_rate": 0.0,
        "avg_alignment_score": 0.0,
        "avg_edit_distance": 0.0,
        "avg_indel_count": 0.0,
        "avg_snp_count": 0.0,
        "avg_total_variants": 0.0,
    }

    for episode in range(n_episodes):
        obs, reset_info = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        
        # Ensure action is numpy array
        action = np.asarray(action, dtype=np.int32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        pattern_str = info["pattern"]
        
        # Extract dataset size information
        dataset_features = reset_info.get("dataset_features", {})
        num_reads_normalized = dataset_features.get("num_reads", 0.0)
        num_reads_actual = num_reads_normalized * 10000.0  # Denormalize
        chromosome = reset_info.get("chromosome", "unknown")
        
        episode_data = {
            "episode": episode + 1,
            "pattern": pattern_str,
            "reward": reward,
            "runtime": info["runtime"],
            "mapping_rate": info["mapping_rate"],
            "alignment_score": info["alignment_score"],
            "edit_distance": info["edit_distance"],
            "indel_count": info.get("indel_count", 0.0),
            "snp_count": info.get("snp_count", 0.0),
            "total_variants": info.get("total_variants", 0.0),
            "num_reads": num_reads_actual,
            "chromosome": chromosome,
        }
        
        results["episodes"].append(episode_data)
        results["total_reward"] += reward
        results["pattern_counts"][pattern_str] += 1
        results["avg_runtime"] += info["runtime"]
        results["avg_mapping_rate"] += info["mapping_rate"]
        results["avg_alignment_score"] += info["alignment_score"]
        results["avg_edit_distance"] += info["edit_distance"]
        results["avg_indel_count"] += info.get("indel_count", 0.0)
        results["avg_snp_count"] += info.get("snp_count", 0.0)
        results["avg_total_variants"] += info.get("total_variants", 0.0)
        
        # Truth-based metrics (if available)
        if "true_snps_tp" in info:
            if "avg_true_snps_tp" not in results:
                results["avg_true_snps_tp"] = 0.0
                results["avg_true_indels_tp"] = 0.0
                results["avg_false_positives"] = 0.0
                results["avg_precision"] = 0.0
                results["avg_recall"] = 0.0
                results["avg_f1_score"] = 0.0
            results["avg_true_snps_tp"] += info.get("true_snps_tp", 0.0)
            results["avg_true_indels_tp"] += info.get("true_indels_tp", 0.0)
            results["avg_false_positives"] += info.get("false_positives", 0.0)
            results["avg_precision"] += info.get("precision", 0.0)
            results["avg_recall"] += info.get("recall", 0.0)
            results["avg_f1_score"] += info.get("f1_score", 0.0)

    # Calculate averages
    results["avg_reward"] = results["total_reward"] / n_episodes
    results["avg_runtime"] /= n_episodes
    results["avg_mapping_rate"] /= n_episodes
    results["avg_alignment_score"] /= n_episodes
    results["avg_edit_distance"] /= n_episodes
    results["avg_indel_count"] /= n_episodes
    results["avg_snp_count"] /= n_episodes
    results["avg_total_variants"] /= n_episodes
    
    # Average truth-based metrics if they exist
    if "avg_true_snps_tp" in results:
        results["avg_true_snps_tp"] /= n_episodes
        results["avg_true_indels_tp"] /= n_episodes
        results["avg_false_positives"] /= n_episodes
        results["avg_precision"] /= n_episodes
        results["avg_recall"] /= n_episodes
        results["avg_f1_score"] /= n_episodes

    return results


def evaluate_baseline(
    env,
    n_episodes: int,
) -> Dict[str, Any]:
    """Evaluate baseline using pattern '10' (50% sparsification) for comparison."""
    results = {
        "episodes": [],
        "total_reward": 0.0,
        "pattern_counts": Counter(),  # Count unique patterns
        "avg_runtime": 0.0,
        "avg_mapping_rate": 0.0,
        "avg_alignment_score": 0.0,
        "avg_edit_distance": 0.0,
        "avg_indel_count": 0.0,
        "avg_snp_count": 0.0,
        "avg_total_variants": 0.0,
    }

    if hasattr(env.action_space, 'nvec'):
        pattern_length = len(env.action_space.nvec)
    else:
        pattern_length = 6

    action = np.zeros(pattern_length, dtype=np.int32)
    action[0] = 1
    action[1] = 0
    action[2] = 1

    for episode in range(n_episodes):
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
        
        pattern_str = info["pattern"]
        
        episode_data = {
            "episode": episode + 1,
            "pattern": pattern_str,
            "reward": reward,
            "runtime": info["runtime"],
            "mapping_rate": info["mapping_rate"],
            "alignment_score": info["alignment_score"],
            "edit_distance": info["edit_distance"],
            "indel_count": info.get("indel_count", 0.0),
            "snp_count": info.get("snp_count", 0.0),
            "total_variants": info.get("total_variants", 0.0),
        }
        
        results["episodes"].append(episode_data)
        results["total_reward"] += reward
        results["pattern_counts"][pattern_str] += 1
        results["avg_runtime"] += info["runtime"]
        results["avg_mapping_rate"] += info["mapping_rate"]
        results["avg_alignment_score"] += info["alignment_score"]
        results["avg_edit_distance"] += info["edit_distance"]
        results["avg_indel_count"] += info.get("indel_count", 0.0)
        results["avg_snp_count"] += info.get("snp_count", 0.0)
        results["avg_total_variants"] += info.get("total_variants", 0.0)
        
        # Truth-based metrics (if available)
        if "true_snps_tp" in info:
            if "avg_true_snps_tp" not in results:
                results["avg_true_snps_tp"] = 0.0
                results["avg_true_indels_tp"] = 0.0
                results["avg_false_positives"] = 0.0
                results["avg_precision"] = 0.0
                results["avg_recall"] = 0.0
                results["avg_f1_score"] = 0.0
            results["avg_true_snps_tp"] += info.get("true_snps_tp", 0.0)
            results["avg_true_indels_tp"] += info.get("true_indels_tp", 0.0)
            results["avg_false_positives"] += info.get("false_positives", 0.0)
            results["avg_precision"] += info.get("precision", 0.0)
            results["avg_recall"] += info.get("recall", 0.0)
            results["avg_f1_score"] += info.get("f1_score", 0.0)

    # Calculate averages
    results["avg_reward"] = results["total_reward"] / n_episodes
    results["avg_runtime"] /= n_episodes
    results["avg_mapping_rate"] /= n_episodes
    results["avg_alignment_score"] /= n_episodes
    results["avg_edit_distance"] /= n_episodes
    results["avg_indel_count"] /= n_episodes
    results["avg_snp_count"] /= n_episodes
    results["avg_total_variants"] /= n_episodes
    
    # Average truth-based metrics if they exist
    if "avg_true_snps_tp" in results:
        results["avg_true_snps_tp"] /= n_episodes
        results["avg_true_indels_tp"] /= n_episodes
        results["avg_false_positives"] /= n_episodes
        results["avg_precision"] /= n_episodes
        results["avg_recall"] /= n_episodes
        results["avg_f1_score"] /= n_episodes

    return results


def print_results(results: Dict[str, Any], title: str):
    """Print evaluation results in a readable format."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    print(f"\nAverage Metrics ({len(results['episodes'])} episodes):")
    print(f"  Reward:           {results['avg_reward']:.4f}")
    print(f"  Runtime:          {results['avg_runtime']:.2f}s")
    print(f"  Mapping Rate:     {results['avg_mapping_rate']:.4f} ({results['avg_mapping_rate']*100:.2f}%)")
    print(f"  Alignment Score:  {results['avg_alignment_score']:.2f}")
    print(f"  Edit Distance:    {results['avg_edit_distance']:.2f}")
    
    # Variant detection metrics
    if 'avg_true_snps_tp' in results:
        # Truth-based metrics
        print(f"\nTruth-Based Variant Metrics:")
        print(f"  True SNPs (TP):     {results['avg_true_snps_tp']:.0f}")
        print(f"  True Indels (TP):   {results['avg_true_indels_tp']:.0f}")
        print(f"  False Positives:    {results['avg_false_positives']:.0f}")
        print(f"  Precision:          {results['avg_precision']:.4f}")
        print(f"  Recall:             {results['avg_recall']:.4f}")
        print(f"  F1 Score:           {results['avg_f1_score']:.4f}")
    else:
        # Raw counts (fallback)
        if 'avg_indel_count' in results:
            print(f"  Indels Detected:  {results['avg_indel_count']:.0f}")
        if 'avg_snp_count' in results:
            print(f"  SNPs Detected:    {results['avg_snp_count']:.0f}")
        if 'avg_total_variants' in results:
            print(f"  Total Variants:   {results['avg_total_variants']:.0f}")
    
    print(f"\nPattern Selection (top 10 most frequent):")
    sorted_patterns = sorted(results["pattern_counts"].items(), key=lambda x: x[1], reverse=True)
    for pattern_str, count in sorted_patterns[:10]:
        percentage = (count / len(results["episodes"])) * 100
        sparsification = (pattern_str.count('0') / len(pattern_str)) * 100 if pattern_str else 0
        print(f"  Pattern '{pattern_str}': {count:3d} times ({percentage:5.1f}%) [sparsification: {sparsification:.1f}%]")
    
    # Analyze pattern selection by dataset size
    if any("num_reads" in ep for ep in results["episodes"]):
        print(f"\nPattern Selection by Dataset Size:")
        pattern_by_size = {}
        for ep in results["episodes"]:
            pattern = ep["pattern"]
            num_reads = ep.get("num_reads", 0.0)
            if pattern not in pattern_by_size:
                pattern_by_size[pattern] = []
            pattern_by_size[pattern].append(num_reads)
        
        for pattern_str in sorted(pattern_by_size.keys()):
            reads_list = pattern_by_size[pattern_str]
            avg_reads = np.mean(reads_list)
            min_reads = np.min(reads_list)
            max_reads = np.max(reads_list)
            print(f"  Pattern '{pattern_str}': avg {avg_reads:.0f} reads (range: {min_reads:.0f}-{max_reads:.0f})")
    
    print(f"\nEpisode Details:")
    for ep in results["episodes"]:
        num_reads = ep.get("num_reads", 0.0)
        chromosome = ep.get("chromosome", "unknown")
        print(
            f"  Ep {ep['episode']:2d}: Pattern '{ep['pattern']}' | "
            f"Reward: {ep['reward']:7.4f} | "
            f"Runtime: {ep['runtime']:5.2f}s | "
            f"MapRate: {ep['mapping_rate']*100:5.1f}% | "
            f"Reads: {num_reads:.0f} | "
            f"Chr: {chromosome}"
        )


def main() -> None:
    args = parse_args()
    
    if not args.model_path.exists():
        print(f"Error: Model not found at {args.model_path}")
        print("Train a model first with: python3 rl/ppo_train.py --timesteps 100")
        return
    
    config_dict = load_config(args.config)
    
    # Determine project root
    project_root = args.config.resolve().parent.parent
    if not (project_root / "GDiet-ShortReads").exists():
        project_root = Path.cwd()
    
    env = load_env_from_config(config_dict, project_root=project_root)
    env.set_evaluation_mode(evaluation_mode=True)
    env = Monitor(env, str(args.model_path.parent / "logs"))
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = PPO.load(str(args.model_path), env=env)
    print("✓ Model loaded successfully")
    
    print(f"\nEvaluating model on {args.n_episodes} episodes...")
    model_results = evaluate_model(model, env, args.n_episodes)
    print_results(model_results, "Trained Model Results")
    
    if args.compare_baseline:
        print(f"\nEvaluating random baseline on {args.n_episodes} episodes...")
        baseline_env = env.env if hasattr(env, 'env') else env
        if hasattr(baseline_env, 'set_evaluation_mode'):
            baseline_env.set_evaluation_mode(evaluation_mode=True)
        baseline_results = evaluate_baseline(env, args.n_episodes)
        print_results(baseline_results, "Random Baseline Results")
        print(f"\n{'='*60}")
        print("Comparison: Trained Model vs Baseline (pattern '10')")
        print(f"{'='*60}")
        print(f"Reward improvement:     {model_results['avg_reward'] - baseline_results['avg_reward']:+.4f}")
        print(f"Runtime improvement:     {baseline_results['avg_runtime'] - model_results['avg_runtime']:+.2f}s ({(baseline_results['avg_runtime']/model_results['avg_runtime'] - 1)*100:+.1f}%)")
        print(f"Mapping rate improvement: {(model_results['avg_mapping_rate'] - baseline_results['avg_mapping_rate'])*100:+.2f}%")
        print(f"Edit distance improvement: {baseline_results['avg_edit_distance'] - model_results['avg_edit_distance']:+.2f}")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
