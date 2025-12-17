"""
Statistical Analysis for Personality Benchmark Results

Performs:
- Descriptive statistics per personality
- Normality tests (Shapiro-Wilk)
- ANOVA or Kruskal-Wallis for group differences
- Posthoc pairwise comparisons with corrections
- Effect sizes (Cohen's d, Cliff's delta)
- Generates summary report
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_results(csv_path):
    """Load benchmark results CSV."""
    return pd.read_csv(csv_path)


def descriptive_stats(df, group_col='personality_id', metrics=None):
    """Compute descriptive statistics per personality."""
    if metrics is None:
        metrics = ['generation_time', 'test_count', 'assertion_count', 
                  'statement_coverage', 'mutation_score', 'valid', 'caught_bug']
    
    stats_dict = {}
    for metric in metrics:
        if metric in df.columns:
            # For boolean metrics, convert to int/float for mean calculation
            if df[metric].dtype == bool:
                df[metric] = df[metric].astype(int)
            
            grouped = df.groupby(group_col)[metric].agg(['mean', 'std', 'median', 'min', 'max', 'count'])
            stats_dict[metric] = grouped
    
    return stats_dict


def test_normality(df, metric, group_col='personality_id'):
    """Test normality per group using Shapiro-Wilk."""
    results = {}
    for group in df[group_col].unique():
        data = df[df[group_col] == group][metric].dropna()
        if len(data) >= 3:
            stat, p = stats.shapiro(data)
            results[group] = {'statistic': stat, 'p_value': p, 'normal': p > 0.05}
        else:
            results[group] = {'statistic': None, 'p_value': None, 'normal': None}
    return results


def anova_or_kruskal(df, metric, group_col='personality_id'):
    """
    Perform ANOVA (if normal) or Kruskal-Wallis (if non-normal).
    
    Returns:
        dict with test name, statistic, p-value
    """
    groups = [df[df[group_col] == g][metric].dropna().values 
              for g in df[group_col].unique()]
    
    # Filter out empty groups
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        return {'test': 'none', 'statistic': None, 'p_value': None}
    
    # Check normality for all groups
    normality = test_normality(df, metric, group_col)
    all_normal = all(v['normal'] for v in normality.values() if v['normal'] is not None)
    
    if all_normal:
        # ANOVA
        stat, p = stats.f_oneway(*groups)
        return {'test': 'ANOVA', 'statistic': stat, 'p_value': p}
    else:
        # Kruskal-Wallis
        stat, p = stats.kruskal(*groups)
        return {'test': 'Kruskal-Wallis', 'statistic': stat, 'p_value': p}


def pairwise_comparisons(df, metric, group_col='personality_id', method='tukey'):
    """
    Perform pairwise posthoc comparisons.
    
    Returns:
        DataFrame with pairwise results
    """
    from scipy.stats import mannwhitneyu, ttest_ind
    from itertools import combinations
    
    groups = df[group_col].unique()
    results = []
    
    for g1, g2 in combinations(groups, 2):
        data1 = df[df[group_col] == g1][metric].dropna().values
        data2 = df[df[group_col] == g2][metric].dropna().values
        
        if len(data1) < 2 or len(data2) < 2:
            continue
        
        # Use Mann-Whitney U for non-parametric
        stat, p = mannwhitneyu(data1, data2, alternative='two-sided')
        
        # Cohen's d effect size
        pooled_std = np.sqrt((np.std(data1, ddof=1)**2 + np.std(data2, ddof=1)**2) / 2)
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
        
        results.append({
            'group1': g1,
            'group2': g2,
            'mean1': np.mean(data1),
            'mean2': np.mean(data2),
            'statistic': stat,
            'p_value': p,
            'cohens_d': cohens_d
        })
    
    results_df = pd.DataFrame(results)
    
    # Holm-Bonferroni correction
    if len(results_df) > 0:
        results_df = results_df.sort_values('p_value')
        m = len(results_df)
        results_df['p_corrected'] = [min(p * (m - i), 1.0) for i, p in enumerate(results_df['p_value'])]
        results_df['significant'] = results_df['p_corrected'] < 0.05
    
    return results_df


def generate_report(df, output_path):
    """Generate a comprehensive text report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PERSONALITY IMPACT ON QA AGENT PERFORMANCE - RESEARCH REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Total runs: {len(df)}\n")
        f.write(f"Personalities: {df['personality_id'].nunique()}\n")
        f.write(f"Tasks: {df['task_id'].nunique() if 'task_id' in df.columns else 'N/A'}\n")
        f.write(f"Seeds per condition: {df.groupby(['personality_id', 'task_id']).size().mean():.1f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("PERSONALITY DESCRIPTIONS (IPIP-NEO inspired)\n")
        f.write("-" * 80 + "\n\n")
        
        # Load and write personality descriptions
        try:
            import yaml
            with open('src/engineering_team/config/personalities.yaml', 'r') as pf:
                personalities = yaml.safe_load(pf)['personalities']
            for p in personalities:
                f.write(f"{p['label']} ({p['id']}):\n")
                f.write(f"  {p['description']}\n\n")
        except Exception as e:
            f.write(f"Could not load personality descriptions: {e}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("DESCRIPTIVE STATISTICS\n")
        f.write("-" * 80 + "\n\n")
        
        metrics = ['generation_time', 'test_count', 'assertion_count', 
                  'statement_coverage', 'mutation_score', 'valid', 'caught_bug']
        
        desc_stats = descriptive_stats(df, metrics=metrics)
        
        for metric, stats_df in desc_stats.items():
            f.write(f"\n{metric.upper().replace('_', ' ')}:\n")
            f.write(stats_df.to_string())
            f.write("\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("HYPOTHESIS TESTS\n")
        f.write("-" * 80 + "\n\n")
        
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            f.write(f"\n{metric.upper().replace('_', ' ')}:\n")
            
            # Normality
            norm_results = test_normality(df, metric)
            f.write("  Normality (Shapiro-Wilk, p > 0.05 = normal):\n")
            for group, result in norm_results.items():
                if result['p_value'] is not None:
                    f.write(f"    {group}: p={result['p_value']:.4f} {'(normal)' if result['normal'] else '(non-normal)'}\n")
            
            # Overall test
            test_result = anova_or_kruskal(df, metric)
            f.write(f"\n  {test_result['test']}:\n")
            
            if test_result['statistic'] is not None:
                f.write(f"    Statistic: {test_result['statistic']:.4f}\n")
            else:
                f.write(f"    Statistic: N/A (insufficient data)\n")
                
            if test_result['p_value'] is not None:
                f.write(f"    P-value: {test_result['p_value']:.4f}\n")
                f.write(f"    Significant: {'YES' if test_result['p_value'] < 0.05 else 'NO'}\n")
            else:
                f.write(f"    P-value: N/A (insufficient data)\n")
                f.write(f"    Significant: N/A\n")
            
            if test_result['p_value'] is not None and test_result['p_value'] < 0.05:
                f.write("\n  Posthoc pairwise comparisons (Mann-Whitney U with Holm-Bonferroni correction):\n")
                pairwise_df = pairwise_comparisons(df, metric)
                sig_pairs = pairwise_df[pairwise_df['significant']]
                if len(sig_pairs) > 0:
                    f.write(sig_pairs[['group1', 'group2', 'mean1', 'mean2', 'cohens_d', 'p_corrected']].to_string(index=False))
                    f.write("\n")
                else:
                    f.write("    No significant pairwise differences after correction.\n")
            
            f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("INTERPRETATION & FINDINGS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("Key observations:\n\n")
        
        # Automated interpretation
        for metric in ['mutation_score', 'statement_coverage', 'test_count']:
            if metric in df.columns:
                test_result = anova_or_kruskal(df, metric)
                if test_result['p_value'] and test_result['p_value'] < 0.05:
                    best_group = df.groupby('personality_id')[metric].mean().idxmax()
                    best_mean = df.groupby('personality_id')[metric].mean().max()
                    f.write(f"- {metric.replace('_', ' ').title()}: Significant differences found (p={test_result['p_value']:.4f}).\n")
                    f.write(f"  Best performer: {best_group} (mean={best_mean:.2f})\n\n")
                else:
                    f.write(f"- {metric.replace('_', ' ').title()}: No significant differences across personalities.\n\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Report generated: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('--input', type=str, default='bench/runs/*/results/*.csv',
                       help='Path to results CSV (supports glob)')
    parser.add_argument('--output', type=str, default='bench/report.txt',
                       help='Output report path')
    
    args = parser.parse_args()
    
    # Find latest results
    from glob import glob
    csv_files = glob(args.input)
    if not csv_files:
        print(f"No results found matching: {args.input}")
        return
    
    latest_csv = max(csv_files, key=lambda p: Path(p).stat().st_mtime)
    print(f"Loading results from: {latest_csv}")
    
    df = load_results(latest_csv)
    print(f"Loaded {len(df)} runs")
    
    generate_report(df, args.output)
    print(f"Analysis complete! Report: {args.output}")


if __name__ == '__main__':
    main()
