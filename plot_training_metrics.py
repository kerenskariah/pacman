"""
Comprehensive Training Metrics Visualization Script for PPO Agent
Generates publication-quality plots for all metrics in training_log.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def load_data(csv_path):
    """Load training log CSV and return DataFrame"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} training updates")
    print(f"Columns: {list(df.columns)}")
    return df


def smooth_curve(data, window=50):
    """Apply moving average smoothing"""
    if len(data) < window:
        return data
    return pd.Series(data).rolling(window=window, min_periods=1).mean()


def plot_score_progression(df, output_dir, smooth_window=50):
    """Plot average score over training updates"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Raw scores
    ax.plot(df['update'], df['avg_score'], alpha=0.3, color='blue', 
            linewidth=0.8, label='Raw Score')
    
    # Smoothed scores
    smoothed = smooth_curve(df['avg_score'], window=smooth_window)
    ax.plot(df['update'], smoothed, color='darkblue', 
            linewidth=2.5, label=f'Smoothed (window={smooth_window})')
    
    # Add milestone annotations
    milestones = [
        (df['update'].min(), df['avg_score'].iloc[0], 'Start'),
        (df['update'].max(), df['avg_score'].iloc[-1], 'End')
    ]
    
    # Peak
    peak_idx = df['avg_score'].idxmax()
    peak_update = df.loc[peak_idx, 'update']
    peak_score = df.loc[peak_idx, 'avg_score']
    milestones.append((peak_update, peak_score, f'Peak: {peak_score:.0f}'))
    
    for update, score, label in milestones:
        ax.scatter(update, score, s=100, c='red', zorder=5, marker='o')
        ax.annotate(label, xy=(update, score), xytext=(10, 10),
                   textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.set_xlabel('Training Update', fontweight='bold')
    ax.set_ylabel('Average Score', fontweight='bold')
    ax.set_title('PPO Agent Score Progression on Ms. Pac-Man', fontweight='bold', fontsize=16)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'score_progression.png', dpi=300)
    plt.close()
    print("✓ Generated: score_progression.png")


def plot_reward_vs_score(df, output_dir):
    """Plot both reward and score on same axes"""
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Training Update', fontweight='bold')
    ax1.set_ylabel('Average Reward (scaled)', fontweight='bold', color=color1)
    ax1.plot(df['update'], df['avg_reward'], color=color1, alpha=0.7, linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Average Score (unscaled)', fontweight='bold', color=color2)
    ax2.plot(df['update'], df['avg_score'], color=color2, alpha=0.7, linewidth=1.5)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Reward vs Score Progression', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_vs_score.png', dpi=300)
    plt.close()
    print("✓ Generated: reward_vs_score.png")


def plot_loss_metrics(df, output_dir, smooth_window=50):
    """Plot policy loss and value loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Policy Loss
    ax1.plot(df['update'], df['policy_loss'], alpha=0.3, color='green', linewidth=0.8)
    smoothed_policy = smooth_curve(df['policy_loss'], window=smooth_window)
    ax1.plot(df['update'], smoothed_policy, color='darkgreen', linewidth=2.5,
            label=f'Smoothed (window={smooth_window})')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_xlabel('Training Update', fontweight='bold')
    ax1.set_ylabel('Policy Loss', fontweight='bold')
    ax1.set_title('Policy Loss Over Training', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Value Loss
    ax2.plot(df['update'], df['value_loss'], alpha=0.3, color='purple', linewidth=0.8)
    smoothed_value = smooth_curve(df['value_loss'], window=smooth_window)
    ax2.plot(df['update'], smoothed_value, color='indigo', linewidth=2.5,
            label=f'Smoothed (window={smooth_window})')
    ax2.set_xlabel('Training Update', fontweight='bold')
    ax2.set_ylabel('Value Loss', fontweight='bold')
    ax2.set_title('Value Loss Over Training', fontweight='bold', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_metrics.png', dpi=300)
    plt.close()
    print("✓ Generated: loss_metrics.png")


def plot_exploration_metrics(df, output_dir, smooth_window=50):
    """Plot entropy, action diversity, and unique states"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Entropy
    ax = axes[0, 0]
    ax.plot(df['update'], df['entropy'], alpha=0.3, color='orange', linewidth=0.8)
    smoothed = smooth_curve(df['entropy'], window=smooth_window)
    ax.plot(df['update'], smoothed, color='darkorange', linewidth=2.5,
           label=f'Smoothed (window={smooth_window})')
    ax.axhline(y=np.log(9), color='red', linestyle='--', alpha=0.5,
              linewidth=2, label=f'Max Entropy (ln(9)={np.log(9):.2f})')
    ax.set_xlabel('Training Update', fontweight='bold')
    ax.set_ylabel('Policy Entropy', fontweight='bold')
    ax.set_title('Policy Entropy (Exploration Indicator)', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Action Diversity
    ax = axes[0, 1]
    ax.plot(df['update'], df['action_diversity'], alpha=0.3, color='teal', linewidth=0.8)
    smoothed = smooth_curve(df['action_diversity'], window=smooth_window)
    ax.plot(df['update'], smoothed, color='darkcyan', linewidth=2.5,
           label=f'Smoothed (window={smooth_window})')
    ax.axhline(y=2.197, color='red', linestyle='--', alpha=0.5,
              linewidth=2, label='Max Diversity (2.197)')
    ax.set_xlabel('Training Update', fontweight='bold')
    ax.set_ylabel('Action Diversity Score', fontweight='bold')
    ax.set_title('Action Diversity (9 Actions)', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Unique States
    ax = axes[1, 0]
    ax.plot(df['update'], df['unique_states'], alpha=0.3, color='brown', linewidth=0.8)
    smoothed = smooth_curve(df['unique_states'], window=smooth_window)
    ax.plot(df['update'], smoothed, color='maroon', linewidth=2.5,
           label=f'Smoothed (window={smooth_window})')
    ax.set_xlabel('Training Update', fontweight='bold')
    ax.set_ylabel('Unique States (window)', fontweight='bold')
    ax.set_title('State Space Exploration', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Clip Fraction
    ax = axes[1, 1]
    ax.plot(df['update'], df['clip_frac'], alpha=0.3, color='magenta', linewidth=0.8)
    smoothed = smooth_curve(df['clip_frac'], window=smooth_window)
    ax.plot(df['update'], smoothed, color='darkmagenta', linewidth=2.5,
           label=f'Smoothed (window={smooth_window})')
    ax.set_xlabel('Training Update', fontweight='bold')
    ax.set_ylabel('Clip Fraction', fontweight='bold')
    ax.set_title('PPO Clipping Fraction', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'exploration_metrics.png', dpi=300)
    plt.close()
    print("✓ Generated: exploration_metrics.png")


def plot_value_function_quality(df, output_dir, smooth_window=50):
    """Plot explained variance, advantage std, and positive advantage fraction"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Explained Variance
    ax = axes[0]
    ax.plot(df['update'], df['explained_variance'], alpha=0.3, color='green', linewidth=0.8)
    smoothed = smooth_curve(df['explained_variance'], window=smooth_window)
    ax.plot(df['update'], smoothed, color='darkgreen', linewidth=2.5,
           label=f'Smoothed (window={smooth_window})')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Perfect (1.0)')
    ax.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Excellent (0.9)')
    ax.set_xlabel('Training Update', fontweight='bold')
    ax.set_ylabel('Explained Variance', fontweight='bold')
    ax.set_title('Value Function Quality (Explained Variance)', fontweight='bold', fontsize=14)
    ax.set_ylim([-0.1, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Advantage Std
    ax = axes[1]
    ax.plot(df['update'], df['advantage_std'], alpha=0.3, color='blue', linewidth=0.8)
    smoothed = smooth_curve(df['advantage_std'], window=smooth_window)
    ax.plot(df['update'], smoothed, color='darkblue', linewidth=2.5,
           label=f'Smoothed (window={smooth_window})')
    ax.set_xlabel('Training Update', fontweight='bold')
    ax.set_ylabel('Advantage Std Dev', fontweight='bold')
    ax.set_title('Advantage Estimation (Std Dev)', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Positive Advantage Fraction
    ax = axes[2]
    ax.plot(df['update'], df['positive_adv_frac'], alpha=0.3, color='purple', linewidth=0.8)
    smoothed = smooth_curve(df['positive_adv_frac'], window=smooth_window)
    ax.plot(df['update'], smoothed, color='indigo', linewidth=2.5,
           label=f'Smoothed (window={smooth_window})')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Balanced (0.5)')
    ax.set_xlabel('Training Update', fontweight='bold')
    ax.set_ylabel('Positive Advantage Fraction', fontweight='bold')
    ax.set_title('Advantage Balance (% Positive)', fontweight='bold', fontsize=14)
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'value_function_quality.png', dpi=300)
    plt.close()
    print("✓ Generated: value_function_quality.png")


def plot_kl_divergence(df, output_dir, smooth_window=50):
    """Plot KL divergence over training"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(df['update'], df['kl'], alpha=0.3, color='red', linewidth=0.8)
    smoothed = smooth_curve(df['kl'], window=smooth_window)
    ax.plot(df['update'], smoothed, color='darkred', linewidth=2.5,
           label=f'Smoothed (window={smooth_window})')
    
    ax.axhline(y=0.01, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Target KL (~0.01)')
    ax.axhline(y=0.03, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='High KL (0.03)')
    
    ax.set_xlabel('Training Update', fontweight='bold')
    ax.set_ylabel('KL Divergence', fontweight='bold')
    ax.set_title('KL Divergence Between Old and New Policy', fontweight='bold', fontsize=16)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'kl_divergence.png', dpi=300)
    plt.close()
    print("✓ Generated: kl_divergence.png")


def plot_comprehensive_dashboard(df, output_dir, smooth_window=50):
    """Create a comprehensive dashboard with all key metrics"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Score Progression
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df['update'], df['avg_score'], alpha=0.3, color='blue', linewidth=0.8)
    smoothed = smooth_curve(df['avg_score'], window=smooth_window)
    ax1.plot(df['update'], smoothed, color='darkblue', linewidth=2.5)
    ax1.set_ylabel('Average Score', fontweight='bold')
    ax1.set_title('Score Progression', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Explained Variance
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(df['update'], df['explained_variance'], alpha=0.3, color='green', linewidth=0.8)
    smoothed = smooth_curve(df['explained_variance'], window=smooth_window)
    ax2.plot(df['update'], smoothed, color='darkgreen', linewidth=2.5)
    ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Explained Var', fontweight='bold')
    ax2.set_title('Value Function', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Policy Loss
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df['update'], df['policy_loss'], alpha=0.3, color='green', linewidth=0.8)
    smoothed = smooth_curve(df['policy_loss'], window=smooth_window)
    ax3.plot(df['update'], smoothed, color='darkgreen', linewidth=2.5)
    ax3.set_ylabel('Policy Loss', fontweight='bold')
    ax3.set_title('Policy Loss', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Value Loss
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df['update'], df['value_loss'], alpha=0.3, color='purple', linewidth=0.8)
    smoothed = smooth_curve(df['value_loss'], window=smooth_window)
    ax4.plot(df['update'], smoothed, color='indigo', linewidth=2.5)
    ax4.set_ylabel('Value Loss', fontweight='bold')
    ax4.set_title('Value Loss', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Entropy
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(df['update'], df['entropy'], alpha=0.3, color='orange', linewidth=0.8)
    smoothed = smooth_curve(df['entropy'], window=smooth_window)
    ax5.plot(df['update'], smoothed, color='darkorange', linewidth=2.5)
    ax5.axhline(y=np.log(9), color='red', linestyle='--', alpha=0.5)
    ax5.set_ylabel('Entropy', fontweight='bold')
    ax5.set_title('Policy Entropy', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Action Diversity
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.plot(df['update'], df['action_diversity'], alpha=0.3, color='teal', linewidth=0.8)
    smoothed = smooth_curve(df['action_diversity'], window=smooth_window)
    ax6.plot(df['update'], smoothed, color='darkcyan', linewidth=2.5)
    ax6.axhline(y=2.197, color='red', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Update', fontweight='bold')
    ax6.set_ylabel('Action Diversity', fontweight='bold')
    ax6.set_title('Action Diversity', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # KL Divergence
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.plot(df['update'], df['kl'], alpha=0.3, color='red', linewidth=0.8)
    smoothed = smooth_curve(df['kl'], window=smooth_window)
    ax7.plot(df['update'], smoothed, color='darkred', linewidth=2.5)
    ax7.set_xlabel('Update', fontweight='bold')
    ax7.set_ylabel('KL Divergence', fontweight='bold')
    ax7.set_title('KL Divergence', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # Advantage Std
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.plot(df['update'], df['advantage_std'], alpha=0.3, color='blue', linewidth=0.8)
    smoothed = smooth_curve(df['advantage_std'], window=smooth_window)
    ax8.plot(df['update'], smoothed, color='darkblue', linewidth=2.5)
    ax8.set_xlabel('Update', fontweight='bold')
    ax8.set_ylabel('Advantage Std', fontweight='bold')
    ax8.set_title('Advantage Std Dev', fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    fig.suptitle('PPO Training Dashboard - All Metrics', fontsize=18, fontweight='bold')
    
    plt.savefig(output_dir / 'comprehensive_dashboard.png', dpi=300)
    plt.close()
    print("✓ Generated: comprehensive_dashboard.png")


def plot_training_phases(df, output_dir):
    """Identify and visualize distinct training phases"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    smoothed_score = smooth_curve(df['avg_score'], window=100)
    ax.plot(df['update'], smoothed_score, color='darkblue', linewidth=3, label='Average Score')
    
    phases = [
        (0, 1500, 'Phase 1:\nInitial Breakthrough', 'lightblue'),
        (1500, 5000, 'Phase 2:\nConsolidation', 'lightgreen'),
        (5000, 8500, 'Phase 3:\nMajor Breakthrough', 'lightyellow'),
        (8500, df['update'].max(), 'Phase 4:\nElite Performance', 'lightcoral')
    ]
    
    for start, end, label, color in phases:
        mask = (df['update'] >= start) & (df['update'] <= end)
        if mask.any():
            ax.axvspan(start, end, alpha=0.2, color=color, label=label)
    
    ax.set_xlabel('Training Update', fontweight='bold')
    ax.set_ylabel('Average Score (smoothed)', fontweight='bold')
    ax.set_title('Training Phases: From Breakthrough to Mastery', fontweight='bold', fontsize=16)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_phases.png', dpi=300)
    plt.close()
    print("✓ Generated: training_phases.png")


def generate_summary_statistics(df, output_dir):
    """Generate and save summary statistics"""
    stats = {
        'Training Summary': {
            'Total Updates': df['update'].max(),
            'Total Episodes': df['episodes_completed'].max(),
            'Starting Score': f"{df['avg_score'].iloc[0]:.1f}",
            'Final Score': f"{df['avg_score'].iloc[-1]:.1f}",
            'Peak Score': f"{df['avg_score'].max():.1f}",
            'Improvement': f"{(df['avg_score'].iloc[-1] / df['avg_score'].iloc[0]):.2f}x"
        },
        'Final Metrics': {
            'Explained Variance': f"{df['explained_variance'].iloc[-1]:.4f}",
            'Entropy': f"{df['entropy'].iloc[-1]:.4f}",
            'Action Diversity': f"{df['action_diversity'].iloc[-1]:.4f}",
            'Advantage Std': f"{df['advantage_std'].iloc[-1]:.4f}",
            'Positive Adv %': f"{df['positive_adv_frac'].iloc[-1]:.4f}",
            'KL Divergence': f"{df['kl'].iloc[-1]:.6f}"
        },
        'Peak Performance': {
            'Best Explained Variance': f"{df['explained_variance'].max():.4f}",
            'Highest Entropy': f"{df['entropy'].max():.4f}",
            'Highest Action Diversity': f"{df['action_diversity'].max():.4f}"
        }
    }
    
    with open(output_dir / 'training_summary.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PPO TRAINING SUMMARY STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        
        for section, metrics in stats.items():
            f.write(f"{section}:\n")
            f.write("-" * 40 + "\n")
            for key, value in metrics.items():
                f.write(f"  {key:.<35} {value}\n")
            f.write("\n")
    
    print("✓ Generated: training_summary.txt")
    return stats


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive training plots')
    parser.add_argument('--csv', type=str, default='results/ppo/training_log.csv',
                       help='Path to training_log.csv')
    parser.add_argument('--output', type=str, default='results/ppo/plots',
                       help='Output directory for plots')
    parser.add_argument('--smooth', type=int, default=50,
                       help='Smoothing window size')
    parser.add_argument('--skip-dashboard', action='store_true',
                       help='Skip generating comprehensive dashboard')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PPO Training Metrics Visualization")
    print("=" * 60)
    print(f"Input CSV: {args.csv}")
    print(f"Output Dir: {output_dir}")
    print(f"Smoothing Window: {args.smooth}")
    print("=" * 60)
    print()
    
    df = load_data(args.csv)
    print()
    
    print("Generating plots...")
    print("-" * 60)
    
    plot_score_progression(df, output_dir, smooth_window=args.smooth)
    plot_reward_vs_score(df, output_dir)
    plot_loss_metrics(df, output_dir, smooth_window=args.smooth)
    plot_exploration_metrics(df, output_dir, smooth_window=args.smooth)
    plot_value_function_quality(df, output_dir, smooth_window=args.smooth)
    plot_kl_divergence(df, output_dir, smooth_window=args.smooth)
    plot_training_phases(df, output_dir)
    
    if not args.skip_dashboard:
        plot_comprehensive_dashboard(df, output_dir, smooth_window=args.smooth)
    
    generate_summary_statistics(df, output_dir)


if __name__ == '__main__':
    main()
