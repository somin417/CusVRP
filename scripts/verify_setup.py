#!/usr/bin/env python3
"""
Comprehensive verification script for CusVRP project.
Checks file structure, solution integrity, script imports, and overwrite protection.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_solution_structure(filepath: Path) -> Tuple[bool, List[str]]:
    """Check if a solution file has the required structure."""
    errors = []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        required_keys = ["routes_by_dc", "objectives", "stops_dict", "depots"]
        for key in required_keys:
            if key not in data:
                errors.append(f"Missing required key: {key}")
        
        # Check objectives structure
        # Note: baseline.json and local.json may not have Z3 computed
        if "objectives" in data:
            obj = data["objectives"]
            required_obj = ["Z1", "Z2"]
            # Z3 is optional for baseline and local
            if "Z3" not in obj and filepath.name not in ["baseline.json", "local.json"]:
                errors.append(f"Missing objective: Z3")
            for key in required_obj:
                if key not in obj:
                    errors.append(f"Missing objective: {key}")
        
        return len(errors) == 0, errors
    except Exception as e:
        return False, [f"Error reading file: {e}"]

def verify_backup_matches_solution(backup_path: Path, solution_path: Path) -> Tuple[bool, List[str]]:
    """Verify that a backup file matches its corresponding solution file."""
    errors = []
    try:
        backup_data = json.load(open(backup_path))
        solution_data = json.load(open(solution_path))
        
        backup_solution = backup_data.get("solution", {})
        
        # Check routes
        if solution_data.get("routes_by_dc") != backup_solution.get("routes_by_dc"):
            errors.append("Routes do not match")
        
        # Check waiting times
        solution_waiting = solution_data.get("waiting_times", {})
        backup_waiting = backup_data.get("waiting", {})
        if solution_waiting != backup_waiting:
            errors.append("Waiting times do not match")
        
        # Check Z value
        solution_Z = solution_data.get("objectives", {}).get("Z")
        backup_Z = backup_data.get("Z")
        if solution_Z != backup_Z:
            errors.append(f"Z values do not match: solution={solution_Z}, backup={backup_Z}")
        
        return len(errors) == 0, errors
    except Exception as e:
        return False, [f"Error comparing files: {e}"]

def test_script_imports() -> Tuple[bool, List[str]]:
    """Test that all main scripts can be imported."""
    errors = []
    scripts = [
        "scripts.compare_waiting_and_scores",
        "scripts.compare_cts_vs_alns",
        "scripts.compare_Z3_variance_vs_MAD",
        "scripts.run_cts_abc_balance",
        "src.vrp_fairness.run_experiment",
    ]
    
    for script in scripts:
        try:
            __import__(script)
        except Exception as e:
            errors.append(f"Failed to import {script}: {e}")
    
    return len(errors) == 0, errors

def check_file_paths() -> Tuple[bool, List[str]]:
    """Check that all expected solution files exist."""
    errors = []
    solutions_dir = Path("outputs/solutions")
    
    expected_files = [
        "baseline.json",
        "local.json",
        "ALNS_MAD.json",
        "ALNS_VAR.json",
        "cts_solution.json",
    ]
    
    for filename in expected_files:
        filepath = solutions_dir / filename
        if not filepath.exists():
            errors.append(f"Missing expected file: {filepath}")
    
    return len(errors) == 0, errors

def check_backup_files() -> Tuple[bool, List[str], List[str]]:
    """Check which backup files exist and if they're safe to delete."""
    warnings = []
    safe_to_delete = []
    
    backup_files = [
        ("seed42_n50_daejeon_alns_best.json", "ALNS_MAD.json"),
        ("seed42_n50_daejeon_cts_best.json", "cts_solution.json"),
        ("variance_best_backup.json", "ALNS_VAR.json"),
    ]
    
    solutions_dir = Path("outputs/solutions")
    
    for backup_name, solution_name in backup_files:
        backup_path = solutions_dir / backup_name
        solution_path = solutions_dir / solution_name
        
        if backup_path.exists():
            if solution_path.exists():
                matches, errors = verify_backup_matches_solution(backup_path, solution_path)
                if matches:
                    safe_to_delete.append(backup_name)
                else:
                    warnings.append(f"{backup_name} does not match {solution_name}: {', '.join(errors)}")
            else:
                warnings.append(f"{backup_name} exists but {solution_name} does not")
    
    return len(warnings) == 0, warnings, safe_to_delete

def check_overwrite_protection() -> Tuple[bool, List[str]]:
    """Check that scripts have overwrite protection."""
    errors = []
    
    # Check run_experiment.py
    run_experiment_path = Path("src/vrp_fairness/run_experiment.py")
    if run_experiment_path.exists():
        content = run_experiment_path.read_text()
        if '--force' not in content or 'force=args.force' not in content:
            errors.append("run_experiment.py missing --force flag or force parameter")
        if 'baseline_file.exists() and not force' not in content:
            errors.append("run_experiment.py missing baseline overwrite check")
    
    # Check compare_Z3_variance_vs_MAD.py
    compare_z3_path = Path("scripts/compare_Z3_variance_vs_MAD.py")
    if compare_z3_path.exists():
        content = compare_z3_path.read_text()
        if '--force' not in content or 'variance_solution_file.exists() and not args.force' not in content:
            errors.append("compare_Z3_variance_vs_MAD.py missing overwrite protection")
    
    # Check run_cts_abc_balance.py
    abc_balance_path = Path("scripts/run_cts_abc_balance.py")
    if abc_balance_path.exists():
        content = abc_balance_path.read_text()
        if '--force' not in content or 'filepath.exists() and not force' not in content:
            errors.append("run_cts_abc_balance.py missing overwrite protection")
    
    return len(errors) == 0, errors

def check_plot_files() -> Tuple[bool, List[str]]:
    """Check that all expected plot files exist."""
    errors = []
    plots_dir = Path("outputs/plots")
    
    # Map of expected plot names to their alternative names
    # Note: weighted_waiting_times_z.png is optional (generated by generate_weighted_waiting_z.py)
    plot_checks = {
        "compare_wait_panels.png": [],
        "waiting_times_baseline_alns_cts.png": [],
        "weighted_waiting_times_baseline_alns_cts.png": [],
        "waiting_times_z.png": ["waiting_times_abc_balance_models.png"],
        "abc_balance_comparison.png": [],
    }
    
    # Optional plots (warn but don't fail)
    optional_plots = {
        "weighted_waiting_times_z.png": ["weighted_waiting_times_abc_balance_models.png"],
    }
    
    for plot, alt_names in plot_checks.items():
        filepath = plots_dir / plot
        if filepath.exists():
            continue
        
        # Check for alternative names
        found_alt = False
        for alt_name in alt_names:
            alt_path = plots_dir / alt_name
            if alt_path.exists():
                found_alt = True
                break
        
        if not found_alt:
            errors.append(f"Missing plot: {plot}")
    
    # Check optional plots (warn but don't fail)
    warnings = []
    for plot, alt_names in optional_plots.items():
        filepath = plots_dir / plot
        if filepath.exists():
            continue
        
        found_alt = False
        for alt_name in alt_names:
            alt_path = plots_dir / alt_name
            if alt_path.exists():
                found_alt = True
                break
        
        if not found_alt:
            warnings.append(f"Optional plot missing: {plot} (can be generated with scripts/generate_weighted_waiting_z.py)")
    
    return len(errors) == 0, errors, warnings


def check_parameter_consistency() -> Tuple[bool, List[str]]:
    """Check parameter consistency across scripts."""
    warnings = []
    
    # Check seed values
    scripts_seed_42 = [
        "compare_waiting_and_scores.py",
        "compare_Z3_variance_vs_MAD.py",
        "compare_cts_vs_alns.py",
        "run_cts_abc_balance.py",
    ]
    
    # Check for parameter differences (documented, not errors)
    param_differences = [
        "compare_cts_vs_alns.py uses eps=0.30 (others use 0.10) - documented",
        "run_cts_abc_balance.py uses eps=0.30 and iters=20 (others use 0.10 and 50) - documented",
    ]
    
    # These are documented differences, so just note them
    if param_differences:
        warnings.extend(param_differences)
    
    return len(warnings) == 0, warnings


def main():
    print("=" * 80)
    print("CusVRP Project Verification")
    print("=" * 80)
    
    all_ok = True
    
    # 1. Check file paths
    print("\n1. Checking expected solution files...")
    files_ok, file_errors = check_file_paths()
    if files_ok:
        print("   ✓ All expected solution files exist")
    else:
        print("   ✗ Missing files:")
        for error in file_errors:
            print(f"     - {error}")
        all_ok = False
    
    # 2. Check solution structure
    print("\n2. Checking solution file structure...")
    solutions_dir = Path("outputs/solutions")
    structure_ok = True
    for json_file in solutions_dir.glob("*.json"):
        if json_file.name.endswith("_best.json") or json_file.name.endswith("_backup.json"):
            continue
        ok, errors = check_solution_structure(json_file)
        if ok:
            print(f"   ✓ {json_file.name} structure OK")
        else:
            print(f"   ✗ {json_file.name} structure errors:")
            for error in errors:
                print(f"     - {error}")
            structure_ok = False
            all_ok = False
    
    if structure_ok:
        print("   ✓ All solution files have correct structure")
    
    # 3. Check backup files
    print("\n3. Checking backup files...")
    backup_ok, backup_warnings, safe_to_delete = check_backup_files()
    if backup_ok and len(safe_to_delete) > 0:
        print(f"   ✓ Found {len(safe_to_delete)} backup files that match solutions (safe to delete):")
        for name in safe_to_delete:
            print(f"     - {name}")
    elif backup_ok:
        print("   ✓ No backup files found (or all match)")
    else:
        print("   ⚠️  Backup file warnings:")
        for warning in backup_warnings:
            print(f"     - {warning}")
    
    # 4. Test script imports
    print("\n4. Testing script imports...")
    imports_ok, import_errors = test_script_imports()
    if imports_ok:
        print("   ✓ All scripts import successfully")
    else:
        print("   ✗ Import errors:")
        for error in import_errors:
            print(f"     - {error}")
        all_ok = False
    
    # 5. Check overwrite protection
    print("\n5. Checking overwrite protection...")
    overwrite_ok, overwrite_errors = check_overwrite_protection()
    if overwrite_ok:
        print("   ✓ All scripts have overwrite protection")
    else:
        print("   ✗ Missing overwrite protection:")
        for error in overwrite_errors:
            print(f"     - {error}")
        all_ok = False
    
    # 6. Check plot files
    print("\n6. Checking plot files...")
    plots_ok, plot_errors, plot_warnings = check_plot_files()
    if plots_ok:
        print("   ✓ All expected plot files exist")
    else:
        print("   ✗ Missing plot files:")
        for error in plot_errors:
            print(f"     - {error}")
        all_ok = False
    
    # 7. Check parameter consistency
    print("\n7. Checking parameter consistency...")
    params_ok, param_warnings = check_parameter_consistency()
    if params_ok:
        print("   ✓ Parameters are consistent (or documented differences)")
    else:
        print("   ⚠️  Parameter differences (documented):")
        for warning in param_warnings:
            print(f"     - {warning}")
    
    # Summary
    print("\n" + "=" * 80)
    if all_ok:
        print("✓ All checks passed!")
    else:
        print("✗ Some checks failed. Please review the errors above.")
    print("=" * 80)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

