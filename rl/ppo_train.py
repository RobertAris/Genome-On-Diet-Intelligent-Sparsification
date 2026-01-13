"""Entry point for PPO training on the GenomeDietPatternEnv."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:
    raise ImportError(
        "stable-baselines3 is required. Install via `pip install stable-baselines3`"
    ) from exc

# Add rl directory to path
sys.path.append(str(Path(__file__).resolve().parent))
from envs.genome_diet_pattern_env import load_env_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on GenomeDietPatternEnv")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "env_default.yaml",
        help="Path to the environment YAML configuration",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50,
        help="Number of PPO timesteps to train",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path(__file__).parent / "models" / "ppo_genome_diet",
        help="Directory to store trained policies",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=1000,
        help="Evaluate the model every N steps",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=5000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments (use 2-4 for multi-core systems to speed up training)",
    )
    parser.add_argument(
        "--use-subproc",
        action="store_true",
        help="Use SubprocVecEnv instead of DummyVecEnv for true parallelism (recommended for n_envs > 1)",
    )
    parser.add_argument(
        "--load-model",
        type=Path,
        default=None,
        help="Path to existing model to continue training from (e.g., rl/models/ppo_genome_diet.zip)",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    args = parse_args()
    config_dict = load_config(args.config)
    
    # Determine project root (parent of rl directory, or config file's parent's parent)
    project_root = args.config.resolve().parent.parent
    if not (project_root / "GDiet-ShortReads").exists():
        # Try current working directory
        project_root = Path.cwd()
    
    # Create environment factory
    def _make_env():
        env = load_env_from_config(config_dict, project_root=project_root)
        # Wrap with Monitor for logging
        log_dir = args.save_path.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, str(log_dir))
        return env
    
    # Create vectorized environment
    if args.n_envs == 1:
        vec_env = DummyVecEnv([_make_env])
    elif args.use_subproc:
        # True parallelism - each environment runs in separate process
        vec_env = SubprocVecEnv([_make_env for _ in range(args.n_envs)])
    else:
        # Multiple environments but sequential execution
        vec_env = DummyVecEnv([_make_env for _ in range(args.n_envs)])
    
    print(f"Using {args.n_envs} environment(s) ({'parallel' if args.use_subproc and args.n_envs > 1 else 'sequential'})")
    
    # Tensorboard logging (optional - set to None if tensorboard not available)
    try:
        import tensorboard
        tb_log = str(args.save_path.parent / "tensorboard")
    except ImportError:
        tb_log = None
        print("Warning: tensorboard not installed, logging disabled")
    
    # Load existing model or create new one
    if args.load_model is not None:
        load_path = Path(args.load_model)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        print(f"Loading existing model from: {load_path}")
        print("Note: Model will continue training with the same hyperparameters")
        
        # Load the model
        model = PPO.load(str(load_path), env=vec_env, tensorboard_log=tb_log)
        print(f"✓ Model loaded successfully")
    else:
        # Create new PPO model
        # Using a simple MLP policy since observations are low-dimensional
        print("Creating new PPO model...")
        model = PPO(
            "MlpPolicy",  # MlpPolicy supports MultiDiscrete action spaces
            vec_env,
            verbose=1,
            learning_rate=5e-4,  # Increased from 3e-4: faster learning, helps escape plateau
            n_steps=64,  # Steps per environment before updating policy
            batch_size=32,  # Batch size for policy updates
            n_epochs=4,  # Number of optimization epochs per update
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.15,  # Increased from 0.1: more exploration to escape local optimum
            vf_coef=0.75,  # Increased from 0.5: more emphasis on value learning (helps with low explained_variance)
            tensorboard_log=tb_log,
        )
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(args.save_path.parent / "checkpoints"),
        name_prefix="ppo_genome_diet",
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback (using same env for simplicity)
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=str(args.save_path.parent / "best_model"),
        log_path=str(args.save_path.parent / "logs"),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    
    # Train the model
    if args.load_model is not None:
        print(f"Continuing training for {args.timesteps} additional timesteps...")
    else:
        print(f"Starting PPO training for {args.timesteps} timesteps...")
    print(f"Action space: {vec_env.action_space}")
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Pattern length: {config_dict['env'].get('pattern_length', 6)}")
    print()
    
    # Check if progress bar dependencies are available
    try:
        import tqdm
        import rich
        use_progress_bar = True
    except ImportError:
        use_progress_bar = False
        print("Note: Progress bar disabled (tqdm/rich not installed)")
    
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=use_progress_bar,
    )
    
    # Save final model
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.save_path))
    print(f"\nTraining complete! Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
