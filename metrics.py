"""
Metrics computation for QA agent benchmarking.

Computes:
- Mutation score (mutmut)
- Statement and branch coverage (coverage.py)
- Test count and assertion count
- Execution time
"""
import subprocess
import json
import re
import time
import os
import sys
from pathlib import Path
import ast


def count_tests_and_assertions(test_file_path):
    """
    Count test functions and assertions in a test file.
    
    Returns:
        dict: {'test_count': int, 'assertion_count': int}
    """
    try:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove markdown code fences if present
        if '```python' in content:
            content = content.split('```python')[1].split('```')[0]
        elif '```' in content:
            content = content.split('```')[1].split('```')[0]
        
        tree = ast.parse(content)
        
        test_count = 0
        assertion_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                test_count += 1
            elif isinstance(node, ast.Assert):
                assertion_count += 1
            elif isinstance(node, ast.Call):
                # Count unittest assertions
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr.startswith('assert'):
                        assertion_count += 1
        
        return {'test_count': test_count, 'assertion_count': assertion_count}
    except Exception as e:
        print(f"Error parsing test file: {e}")
        return {'test_count': 0, 'assertion_count': 0}


def measure_coverage(module_path, test_file_path, work_dir):
    """
    Measure statement and branch coverage using coverage.py.
    
    Returns:
        dict: {'statement_coverage': float, 'branch_coverage': float} or None on error
    """
    work_dir = Path(work_dir)
    cov_data_file = work_dir / '.coverage'
    
    try:
        # Run tests with coverage
        # Ensure module directory is importable during pytest
        env = os.environ.copy()
        existing_pp = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = f"{str(module_path.parent)}{os.pathsep}{existing_pp}" if existing_pp else str(module_path.parent)

        # Use the same interpreter to run coverage
        cmd = [
            sys.executable, '-m', 'coverage', 'run',
            '--source', str(module_path.parent),
            '--data-file', str(cov_data_file),
            '-m', 'pytest', str(test_file_path), '-v'
        ]
        
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        
        # Get coverage report
        report_cmd = [
            sys.executable, '-m', 'coverage', 'report',
            '--data-file', str(cov_data_file)
        ]
        
        report_result = subprocess.run(
            report_cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Parse coverage percentage
        # Example line: "module.py    20      5    75%"
        lines = report_result.stdout.strip().split('\n')
        target_name = module_path.name
        for line in lines:
            if target_name in line:
                parts = line.split()
                if len(parts) >= 4:
                    coverage_str = parts[-1].rstrip('%')
                    try:
                        statement_coverage = float(coverage_str)
                        return {
                            'statement_coverage': statement_coverage,
                            'branch_coverage': None  # coverage.py branch requires --branch flag
                        }
                    except ValueError:
                        pass
        
        return {'statement_coverage': 0.0, 'branch_coverage': None}
    
    except Exception as e:
        print(f"Coverage measurement error: {e}")
        return {'statement_coverage': 0.0, 'branch_coverage': None}


def measure_mutation_score(module_path, test_file_path, work_dir):
    """
    Measure mutation score using mutmut.
    
    Returns:
        dict: {'mutation_score': float, 'killed': int, 'survived': int, 'timeout': int}
    """
    work_dir = Path(work_dir)
    
    try:
        # Run mutmut
        cmd = [
            sys.executable, '-m', 'mutmut', 'run',
            '--paths-to-mutate', str(module_path),
            '--tests-dir', str(test_file_path.parent),
            '--runner', 'pytest'
        ]
        
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes max for mutation testing
        )
        
        # Parse mutmut output
        output = result.stdout + result.stderr
        
        # Example: "killed: 10, survived: 2, timeout: 0"
        killed = survived = timeout = 0
        
        killed_match = re.search(r'killed[:\s]+(\d+)', output, re.IGNORECASE)
        survived_match = re.search(r'survived[:\s]+(\d+)', output, re.IGNORECASE)
        timeout_match = re.search(r'timeout[:\s]+(\d+)', output, re.IGNORECASE)
        
        if killed_match:
            killed = int(killed_match.group(1))
        if survived_match:
            survived = int(survived_match.group(1))
        if timeout_match:
            timeout = int(timeout_match.group(1))
        
        total = killed + survived + timeout
        mutation_score = (killed / total * 100) if total > 0 else 0.0
        
        return {
            'mutation_score': mutation_score,
            'killed': killed,
            'survived': survived,
            'timeout': timeout
        }
    
    except subprocess.TimeoutExpired:
        print("Mutation testing timed out")
        return {'mutation_score': 0.0, 'killed': 0, 'survived': 0, 'timeout': 0}
    except Exception as e:
        print(f"Mutation testing error: {e}")
        return {'mutation_score': 0.0, 'killed': 0, 'survived': 0, 'timeout': 0}


def test_against_buggy_variant(buggy_module_path, test_file_path, work_dir):
    """
    Run tests against a buggy variant to check if tests catch the bug.
    
    Returns:
        dict: {'caught_bug': bool, 'failures': int}
    """
    try:
        # Ensure module directory is importable
        env = os.environ.copy()
        existing_pp = env.get('PYTHONPATH', '')
        # Assuming module_path is like .../modules/task_11.py, we need .../modules in PYTHONPATH
        env['PYTHONPATH'] = f"{str(Path(test_file_path).parent / 'modules')}{os.pathsep}{existing_pp}" if existing_pp else str(Path(test_file_path).parent / 'modules')

        cmd = ['pytest', str(test_file_path), '-v', '--tb=short']
        
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        
        # If tests fail against buggy code, that's good (bug was caught)
        caught_bug = result.returncode != 0
        
        # Count failures
        failures = 0
        if 'failed' in result.stdout.lower():
            match = re.search(r'(\d+)\s+failed', result.stdout)
            if match:
                failures = int(match.group(1))
        
        return {'caught_bug': caught_bug, 'failures': failures}
    
    except Exception as e:
        print(f"Buggy variant test error: {e}")
        return {'caught_bug': False, 'failures': 0}


def check_test_validity(test_file_path, work_dir):
    """
    Check if tests pass against the reference implementation.
    
    Returns:
        dict: {'valid': bool, 'validity_failures': int}
    """
    try:
        # Ensure module directory is importable
        env = os.environ.copy()
        existing_pp = env.get('PYTHONPATH', '')
        # Assuming module_path is like .../modules/task_11.py, we need .../modules in PYTHONPATH
        env['PYTHONPATH'] = f"{str(Path(test_file_path).parent / 'modules')}{os.pathsep}{existing_pp}" if existing_pp else str(Path(test_file_path).parent / 'modules')

        # Run pytest against the reference implementation
        cmd = ['pytest', str(test_file_path), '-v', '--tb=short']
        
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        
        # Tests are valid if they pass (return code 0)
        valid = result.returncode == 0
        
        # Count failures
        failures = 0
        if 'failed' in result.stdout.lower():
            match = re.search(r'(\d+)\s+failed', result.stdout)
            if match:
                failures = int(match.group(1))
        
        return {'valid': valid, 'validity_failures': failures}
    
    except Exception as e:
        print(f"Validity check error: {e}")
        return {'valid': False, 'validity_failures': 0}


def compute_all_metrics(module_path, test_file_path, buggy_module_path=None, work_dir=None, enable_mutation=False):
    """
    Compute all metrics for a generated test file.
    
    Returns:
        dict with all metrics
    """
    if work_dir is None:
        work_dir = test_file_path.parent
    
    metrics = {}
    
    # Basic counts
    counts = count_tests_and_assertions(test_file_path)
    metrics.update(counts)
    
    # Validity check
    validity_result = check_test_validity(test_file_path, work_dir)
    metrics.update(validity_result)
    
    # Coverage
    coverage_result = measure_coverage(module_path, test_file_path, work_dir)
    metrics.update(coverage_result)
    
    # Mutation score
    if enable_mutation:
        mutation_result = measure_mutation_score(module_path, test_file_path, work_dir)
        metrics.update(mutation_result)
    else:
        metrics.update({'mutation_score': None, 'killed': None, 'survived': None, 'timeout': None})
    
    # Buggy variant (if provided)
    if buggy_module_path:
        buggy_result = test_against_buggy_variant(buggy_module_path, test_file_path, work_dir)
        metrics.update(buggy_result)
    
    return metrics
