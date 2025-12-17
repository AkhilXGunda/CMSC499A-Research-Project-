"""
Benchmark Runner: Orchestrates QA agent experiments across personalities and seeds.

Modes:
- pilot: 3 tasks, 3 personalities, 2 seeds (~30 minutes)
- qa-only: 10 tasks, 11 personalities, 3 seeds (QA agent only, isolated)
- full: 5 SWE-bench-lite issues, 11 personalities, 2 seeds (full pipeline)
"""
import sys
print("Runner starting...")
import os
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
import argparse
import yaml
import json
import csv
import time
import random
import traceback
from pathlib import Path
from datetime import datetime

# Add src and bench to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

# from engineering_team.crew import EngineeringTeam
from bench.datasets.mbpp_loader import load_mbpp_tasks, save_reference_implementation, create_buggy_variant
from bench.datasets.complex_loader import load_complex_tasks
from bench.metrics import compute_all_metrics


def load_personalities(config_path):
    """Load personality definitions from YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['personalities']


def run_qa_only_experiment(personality, task, seed, output_dir, token_tracker=None, enable_coverage=True, enable_mutation=False):
    """
    Run QA agent on a single MBPP task with given personality and seed.
    
    Returns:
        dict: metrics and metadata
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save reference implementation
    ref_path = save_reference_implementation(task, output_dir / 'modules')
    buggy_path = create_buggy_variant(task, output_dir / 'modules', mutation_type='boundary')

    # Helper to alias function names so generated tests can import the reference module
    def _ensure_test_imports_resolvable(module_path, test_source, task_id):
        try:
            import ast
            expected_names = set()
            tree = ast.parse(test_source)
            target_mod = f"task_{task_id}"
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == target_mod:
                    for alias in node.names:
                        if alias.name != '*':
                            expected_names.add(alias.asname or alias.name)

            if not expected_names:
                return

            with open(module_path, 'r', encoding='utf-8') as mf:
                mod_src = mf.read()
            mod_tree = ast.parse(mod_src)
            mod_funcs = [n.name for n in mod_tree.body if isinstance(n, ast.FunctionDef)]

            if not mod_funcs:
                return

            existing = set(mod_funcs)
            primary = mod_funcs[0]
            aliases = []
            for name in expected_names:
                if name not in existing:
                    aliases.append(f"{name} = {primary}")

            if aliases:
                with open(module_path, 'a', encoding='utf-8') as mf:
                    mf.write("\n\n# Auto-aliases for benchmark test compatibility\n" + "\n".join(aliases) + "\n")
        except Exception:
            pass
    
    # Prepare inputs for crew
    module_name = f"task_{task['task_id']}.py"
    
    # Create a minimal "requirements" from task description
    requirements = f"""
{task['text']}

Reference solution exists. Your task is to write comprehensive unit tests.
"""
    
    inputs = {
        'requirements': requirements,
        'module_name': module_name,
        'class_name': 'Solution',  # Generic
        'personality': f"{personality['label']}: {personality['description']}"
    }
    
    # Override output to bench artifacts
    original_output = Path('output')
    bench_output = output_dir / 'output'
    bench_output.mkdir(exist_ok=True)
    
    # Time the generation
    start_time = time.time()
    
    try:
        # Run only QA task in isolation
        from crewai import Agent, Task, Crew, Process
        import yaml
        
        # Load agent config
        agents_config_path = Path('src/engineering_team/config/agents.yaml')
        with open(agents_config_path, 'r') as f:
            agents_config = yaml.safe_load(f)
        
        # Create QA agent
        qa_agent = Agent(
            role=agents_config['qa']['role'].format(**inputs),
            goal=agents_config['qa']['goal'].format(**inputs),
            backstory=agents_config['qa']['backstory'].format(**inputs),
            llm=agents_config['qa']['llm'],
            verbose=False
        )
        
        # Create test task
        test_task = Task(
            description=f"Write unit tests for the module at {ref_path}. The module solves: {task['text']}",
            expected_output="A complete test file with unit tests.",
            agent=qa_agent
        )
        
        # Run just the QA task
        mini_crew = Crew(
            agents=[qa_agent],
            tasks=[test_task],
            process=Process.sequential,
            verbose=False
        )
        
        result = mini_crew.kickoff()
        generation_time = time.time() - start_time
        
        # Save generated test code to artifact
        artifact_path = output_dir / 'artifacts' / f"test_task{task['task_id']}_seed{seed}.py"
        artifact_path.parent.mkdir(exist_ok=True)
        
        # Extract code from result (it's a string); strip markdown fences if present
        test_code = str(result)
        if '```' in test_code:
            if '```python' in test_code:
                try:
                    test_code = test_code.split('```python', 1)[1].split('```', 1)[0]
                except Exception:
                    pass
            else:
                try:
                    test_code = test_code.split('```', 1)[1].split('```', 1)[0]
                except Exception:
                    pass
            # Ensure test imports can resolve
            _ensure_test_imports_resolvable(ref_path, test_code, task['task_id'])
            
        artifact_path.write_text(test_code, encoding='utf-8')
        
        # Compute metrics
        metrics = compute_all_metrics(
            module_path=ref_path,
            test_file_path=artifact_path,
            buggy_module_path=buggy_path,
            work_dir=output_dir,
            enable_mutation=enable_mutation
        )
        
        metrics['generation_time'] = generation_time
        metrics['success'] = True
        
        # Token tracking (if available)
        if token_tracker:
            metrics['tokens_input'] = token_tracker.get('input_tokens', 0)
            metrics['tokens_output'] = token_tracker.get('output_tokens', 0)
        
    except Exception as e:
        print(f"Error in experiment: {e}")
        traceback.print_exc()
        with open('error_log.txt', 'a') as f:
            f.write(f"\n\nError in run {personality['id']} task {task['task_id']}:\n")
            f.write(traceback.format_exc())
        
        metrics = {
            'generation_time': time.time() - start_time,
            'success': False,
            'error': str(e),
            'test_count': 0,
            'assertion_count': 0,
            'statement_coverage': 0.0,
            'mutation_score': 0.0
        }
    
    # Add metadata
    result_data = {
        'timestamp': datetime.now().isoformat(),
        'personality_id': personality['id'],
        'personality_label': personality['label'],
        'task_id': task['task_id'],
        'seed': seed,
        'mode': 'qa-only',
        **metrics
    }
    
    return result_data


def run_pilot(output_dir, enable_coverage=True):
    """Run a quick pilot: 3 tasks, 3 personalities, 2 seeds."""
    personalities_path = Path('src/engineering_team/config/personalities.yaml')
    personalities = load_personalities(personalities_path)
    
    # Select 3 personalities: neutral, high conscientiousness, low conscientiousness
    selected = [p for p in personalities if p['id'] in ['neutral', 'conscientiousness_high', 'conscientiousness_low']]
    
    # Load 3 MBPP tasks
    tasks = load_mbpp_tasks(subset_size=3)
    
    seeds = [42, 123]
    
    results = []
    total = len(selected) * len(tasks) * len(seeds)
    counter = 0
    
    print(f"Running pilot: {len(selected)} personalities × {len(tasks)} tasks × {len(seeds)} seeds = {total} runs")
    
    for personality in selected:
        for task in tasks:
            for seed in seeds:
                counter += 1
                print(f"\n[{counter}/{total}] Running: {personality['id']}, task {task['task_id']}, seed {seed}")
                
                result = run_qa_only_experiment(
                    personality=personality,
                    task=task,
                    seed=seed,
                    output_dir=output_dir / f"{personality['id']}_task{task['task_id']}_seed{seed}",
                    enable_coverage=enable_coverage
                )
                
                results.append(result)
                
                # Save incrementally
                save_results(results, output_dir / 'results' / 'pilot_results.csv')
    
    print(f"\nPilot complete. Results saved to {output_dir / 'results' / 'pilot_results.csv'}")
    return results


def run_qa_only_full(output_dir, enable_coverage=True):
    """Run full QA-only benchmark: 10 tasks, 11 personalities, 3 seeds."""
    personalities_path = Path('src/engineering_team/config/personalities.yaml')
    personalities = load_personalities(personalities_path)
    
    tasks = load_mbpp_tasks(subset_size=10)
    seeds = [42, 123, 456]
    
    results = []
    total = len(personalities) * len(tasks) * len(seeds)
    counter = 0
    
    print(f"Running QA-only full: {len(personalities)} personalities × {len(tasks)} tasks × {len(seeds)} seeds = {total} runs")
    
    for personality in personalities:
        print(f"\n=== Personality: {personality['id']} ===")
        for task in tasks:
            for seed in seeds:
                counter += 1
                print(f"[{counter}/{total}] Task {task['task_id']}, seed {seed}")
                
                result = run_qa_only_experiment(
                    personality=personality,
                    task=task,
                    seed=seed,
                    output_dir=output_dir / f"{personality['id']}_task{task['task_id']}_seed{seed}",
                    enable_coverage=enable_coverage
                )
                
                results.append(result)
                
                # Save incrementally every 10 runs
                if counter % 10 == 0:
                    save_results(results, output_dir / 'results' / 'qa_only_results.csv')
    
    save_results(results, output_dir / 'results' / 'qa_only_results.csv')
    print(f"\nQA-only benchmark complete. Results saved to {output_dir / 'results' / 'qa_only_results.csv'}")
    return results


def run_final_benchmark(output_dir, enable_coverage=True, enable_mutation=False):
    """
    Run the final benchmark: 5 personalities x (30 MBPP + 3 Complex) x 2 seeds.
    """
    personalities_list = load_personalities('src/engineering_team/config/personalities.yaml')
    
    # Selected personalities
    selected_ids = [
        'neutral',
        'neuroticism_high',
        'conscientiousness_high',
        'extraversion_high',
        'openness_high'
    ]
    
    # Filter personalities
    target_personalities = [p for p in personalities_list if p['id'] in selected_ids]
    
    # Load tasks
    mbpp_tasks = load_mbpp_tasks(subset_size=30)
    complex_tasks = load_complex_tasks()
    all_tasks = mbpp_tasks + complex_tasks
    
    seeds = [42, 123]
    results = []
    completed_keys = set()
    partial_csv = output_dir / 'results' / 'final_results_partial.csv'
    salvaged_csv = Path('bench/final_results.csv')
    
    if partial_csv.exists():
        print(f"Resuming from {partial_csv}...")
        try:
            with open(partial_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results.append(row)
                    completed_keys.add((row['personality_id'], str(row['task_id']), str(row['seed'])))
            print(f"Loaded {len(results)} completed runs.")
        except Exception as e:
            print(f"Error loading partial results: {e}. Starting fresh.")
            results = []
            completed_keys = set()
    elif salvaged_csv.exists():
        print(f"Seeding from salvaged results: {salvaged_csv}...")
        try:
            with open(salvaged_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results.append(row)
                    completed_keys.add((row['personality_id'], str(row['task_id']), str(row['seed'])))
            print(f"Loaded {len(results)} completed runs from salvage.")
            # Write to new partial file
            save_results(results, partial_csv)
        except Exception as e:
            print(f"Error loading salvaged results: {e}")
            results = []
            completed_keys = set()
    
    total_runs = len(target_personalities) * len(all_tasks) * len(seeds)
    print(f"Plan: {len(target_personalities)} personalities x {len(all_tasks)} tasks x {len(seeds)} seeds = {total_runs} runs")
    
    completed = len(results)
    
    for personality in target_personalities:
        pid = personality['id']
        for task in all_tasks:
            for seed in seeds:
                if (pid, str(task['task_id']), str(seed)) in completed_keys:
                    # print(f"Skipping {pid} on {task['task_id']} (seed {seed}) - already done.")
                    continue

                print(f"[{completed+1}/{total_runs}] MISSING run {pid}-{task['task_id']}-{seed}. Executing...")
                
                metrics = run_qa_only_experiment(
                    personality=personality,
                    task=task,
                    seed=seed,
                    output_dir=output_dir / f"{pid}_{task['task_id']}_seed{seed}",
                    enable_coverage=enable_coverage,
                    enable_mutation=enable_mutation
                )
                
                # Add metadata
                metrics['personality_id'] = pid
                metrics['task_id'] = task['task_id']
                metrics['seed'] = seed
                metrics['task_type'] = 'complex' if 'complex' in str(task['task_id']) else 'mbpp'
                
                results.append(metrics)
                completed += 1
                
                # Save intermediate results
                save_results(results, output_dir / 'results' / 'final_results_partial.csv')
    
    save_results(results, output_dir / 'results' / 'final_results.csv')
    return results


def save_results(results, output_path):
    """Save results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not results:
        return
    
    keys = results[0].keys()
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Saved {len(results)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run QA personality benchmark')
    parser.add_argument('--mode', choices=['pilot', 'qa-only', 'full', 'final'], default='pilot',
                       help='Benchmark mode')
    parser.add_argument('--output', type=str, default='bench/runs',
                       help='Output directory for results')
    parser.add_argument('--no-coverage', action='store_true', help='Disable coverage measurement for speed')
    parser.add_argument('--mutation', action='store_true', help='Enable mutation testing (slow)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output) / datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting benchmark: {args.mode}")
    print(f"Output directory: {output_dir}")
    
    enable_coverage = not args.no_coverage
    enable_mutation = args.mutation

    if args.mode == 'pilot':
        results = run_pilot(output_dir, enable_coverage=enable_coverage)
    elif args.mode == 'qa-only':
        results = run_qa_only_full(output_dir, enable_coverage=enable_coverage)
    elif args.mode == 'final':
        results = run_final_benchmark(output_dir, enable_coverage=enable_coverage, enable_mutation=enable_mutation)
    else:
        print("Full pipeline mode not yet implemented")
        return
    
    print(f"\nBenchmark complete! {len(results)} runs finished.")
    print(f"Results: {output_dir / 'results'}")


if __name__ == '__main__':
    main()
