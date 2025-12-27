#!/usr/bin/env python3
"""
Structure and Logic Validation for 3D Torsion Memory Lattice

Validates code structure, imports, syntax, and logical consistency
without requiring external dependencies (PyTorch, NumPy).
"""

import sys
import ast
import os
from pathlib import Path


class ValidationRunner:
    """Manages validation checks."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.checks = []

    def check(self, name, condition, details=""):
        """Run a validation check."""
        if condition:
            print(f"✅ {name}")
            self.passed += 1
            self.checks.append((name, True))
        else:
            print(f"❌ {name}: {details}")
            self.failed += 1
            self.checks.append((name, False))

    def print_summary(self):
        """Print validation summary."""
        total = self.passed + self.failed
        print(f"\n{'='*80}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total Checks: {total}")
        print(f"Passed: {self.passed} ✅")
        print(f"Failed: {self.failed} ❌")
        print(f"Success Rate: {100 * self.passed / total if total > 0 else 0:.1f}%")
        print(f"{'='*80}\n")


validator = ValidationRunner()

# Base directory
base_dir = Path(__file__).parent.parent
torsion_dir = base_dir / "grcm" / "echozero" / "identity" / "torsion_lattice"

print("="*80)
print("TORSION LATTICE STRUCTURE VALIDATION")
print("="*80)
print(f"Base directory: {base_dir}")
print(f"Torsion directory: {torsion_dir}")
print("="*80 + "\n")


# ============================================================================
# 1. FILE EXISTENCE CHECKS
# ============================================================================

print("Checking file structure...")

files_to_check = [
    ("lattice3d.py", torsion_dir / "lattice3d.py"),
    ("update.py", torsion_dir / "update.py"),
    ("readout.py", torsion_dir / "readout.py"),
    ("writeback.py", torsion_dir / "writeback.py"),
    ("config.yaml", torsion_dir / "config.yaml"),
    ("__init__.py (torsion)", torsion_dir / "__init__.py"),
    ("__init__.py (identity)", torsion_dir.parent / "__init__.py"),
]

for name, path in files_to_check:
    validator.check(f"File exists: {name}", path.exists(), f"Missing: {path}")


# ============================================================================
# 2. SYNTAX VALIDATION
# ============================================================================

print("\nChecking Python syntax...")

python_files = [
    torsion_dir / "lattice3d.py",
    torsion_dir / "update.py",
    torsion_dir / "readout.py",
    torsion_dir / "writeback.py",
    torsion_dir / "__init__.py",
]

for py_file in python_files:
    if py_file.exists():
        try:
            with open(py_file, 'r') as f:
                code = f.read()
            ast.parse(code)
            validator.check(f"Syntax valid: {py_file.name}", True)
        except SyntaxError as e:
            validator.check(f"Syntax valid: {py_file.name}", False, str(e))
    else:
        validator.check(f"Syntax valid: {py_file.name}", False, "File not found")


# ============================================================================
# 3. CLASS AND FUNCTION DEFINITIONS
# ============================================================================

print("\nChecking class definitions...")

def check_class_exists(file_path, class_name):
    """Check if a class is defined in a file."""
    if not file_path.exists():
        return False
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return True
        return False
    except:
        return False

def check_function_exists(file_path, func_name):
    """Check if a function is defined in a file."""
    if not file_path.exists():
        return False
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return True
        return False
    except:
        return False

# Check main classes
validator.check(
    "Class: TorsionLattice3D",
    check_class_exists(torsion_dir / "lattice3d.py", "TorsionLattice3D")
)

validator.check(
    "Class: LatticeUpdater",
    check_class_exists(torsion_dir / "update.py", "LatticeUpdater")
)

# Check key methods in TorsionLattice3D
lattice_methods = [
    "init_checkerboard",
    "step",
    "write_vector",
    "read_vector",
    "get_energy",
    "get_magnetization",
    "reset",
]

for method in lattice_methods:
    validator.check(
        f"Method: TorsionLattice3D.{method}",
        check_function_exists(torsion_dir / "lattice3d.py", method)
    )

# Check key methods in LatticeUpdater
updater_methods = [
    "maybe_sync",
    "force_sync",
    "get_stats",
    "reset",
]

for method in updater_methods:
    validator.check(
        f"Method: LatticeUpdater.{method}",
        check_function_exists(torsion_dir / "update.py", method)
    )

# Check utility functions
validator.check(
    "Function: read_identity",
    check_function_exists(torsion_dir / "readout.py", "read_identity")
)

validator.check(
    "Function: write_identity",
    check_function_exists(torsion_dir / "writeback.py", "write_identity")
)


# ============================================================================
# 4. IMPORT STRUCTURE
# ============================================================================

print("\nChecking import structure...")

def get_imports(file_path):
    """Extract imports from a Python file."""
    if not file_path.exists():
        return []
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        return imports
    except:
        return []

# Check lattice3d.py imports NumPy
lattice_imports = get_imports(torsion_dir / "lattice3d.py")
validator.check(
    "Import: lattice3d.py uses numpy",
    "numpy" in lattice_imports
)

# Check update.py imports from lattice3d
update_file = torsion_dir / "update.py"
if update_file.exists():
    try:
        with open(update_file, 'r') as f:
            content = f.read()
        validator.check(
            "Import: update.py imports TorsionLattice3D",
            "from .lattice3d import TorsionLattice3D" in content or
            "from lattice3d import TorsionLattice3D" in content
        )
    except:
        validator.check("Import: update.py imports TorsionLattice3D", False)


# ============================================================================
# 5. CONFIGURATION FILE
# ============================================================================

print("\nChecking configuration...")

config_file = torsion_dir / "config.yaml"
if config_file.exists():
    try:
        with open(config_file, 'r') as f:
            config_content = f.read()

        # Check for key configuration parameters
        config_keys = [
            "size",
            "coupling_strength",
            "decay_gamma",
            "write_strength",
            "sync_interval_steps",
            "torsion_threshold",
        ]

        for key in config_keys:
            validator.check(
                f"Config key: {key}",
                key in config_content
            )

    except Exception as e:
        validator.check("Config file readable", False, str(e))
else:
    validator.check("Config file exists", False)


# ============================================================================
# 6. MODULE EXPORTS
# ============================================================================

print("\nChecking module exports...")

init_file = torsion_dir / "__init__.py"
if init_file.exists():
    try:
        with open(init_file, 'r') as f:
            init_content = f.read()

        exports_to_check = [
            "TorsionLattice3D",
            "LatticeUpdater",
            "read_identity",
            "write_identity",
        ]

        for export in exports_to_check:
            validator.check(
                f"Export: {export}",
                export in init_content
            )

    except Exception as e:
        validator.check("Module __init__.py readable", False, str(e))


# ============================================================================
# 7. INTEGRATION WITH HYBRID
# ============================================================================

print("\nChecking hybrid integration...")

hybrid_forward = base_dir / "grcm" / "hybrid" / "forward.py"
if hybrid_forward.exists():
    try:
        with open(hybrid_forward, 'r') as f:
            hybrid_content = f.read()

        integration_checks = [
            ("Import TorsionLattice3D", "TorsionLattice3D" in hybrid_content),
            ("Import LatticeUpdater", "LatticeUpdater" in hybrid_content),
            ("Import MobiusEchoLayer", "MobiusEchoLayer" in hybrid_content),
            ("Parameter enable_torsion_lattice", "enable_torsion_lattice" in hybrid_content),
            ("Lattice initialization", "self.torsion_lattice" in hybrid_content),
            ("Updater initialization", "self.lattice_updater" in hybrid_content),
            ("Sync call", "maybe_sync" in hybrid_content),
        ]

        for name, condition in integration_checks:
            validator.check(f"Integration: {name}", condition)

    except Exception as e:
        validator.check("Hybrid forward.py readable", False, str(e))


# ============================================================================
# 8. TEST FILES
# ============================================================================

print("\nChecking test files...")

test_files = [
    base_dir / "tests" / "test_torsion_lattice.py",
    base_dir / "tests" / "test_integration_triads.py",
]

for test_file in test_files:
    validator.check(
        f"Test file: {test_file.name}",
        test_file.exists()
    )

    if test_file.exists():
        try:
            with open(test_file, 'r') as f:
                ast.parse(f.read())
            validator.check(f"Test syntax: {test_file.name}", True)
        except SyntaxError:
            validator.check(f"Test syntax: {test_file.name}", False)


# ============================================================================
# 9. BENCHMARK FILES
# ============================================================================

print("\nChecking benchmark files...")

benchmark_file = base_dir / "benchmarks" / "bench_triads.py"
if benchmark_file.exists():
    validator.check("Benchmark file exists", True)
    try:
        with open(benchmark_file, 'r') as f:
            ast.parse(f.read())
        validator.check("Benchmark syntax valid", True)
    except SyntaxError:
        validator.check("Benchmark syntax valid", False)
else:
    validator.check("Benchmark file exists", False)


# ============================================================================
# 10. LINE COUNT VALIDATION
# ============================================================================

print("\nChecking code metrics...")

def count_lines(file_path):
    """Count non-empty, non-comment lines in a file."""
    if not file_path.exists():
        return 0
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        code_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                code_lines += 1
        return code_lines
    except:
        return 0

module_files = [
    torsion_dir / "lattice3d.py",
    torsion_dir / "update.py",
    torsion_dir / "readout.py",
    torsion_dir / "writeback.py",
]

total_lines = sum(count_lines(f) for f in module_files)
validator.check(
    f"Code volume: {total_lines} lines",
    total_lines > 500,
    f"Only {total_lines} lines found"
)


# ============================================================================
# FINAL SUMMARY
# ============================================================================

validator.print_summary()

if validator.failed == 0:
    print("🎉 ALL VALIDATION CHECKS PASSED! 🎉")
    print("\nThe 3D Torsion Memory Lattice implementation is structurally sound.")
    print("Ready for runtime testing when dependencies (NumPy/PyTorch) are available.\n")
    sys.exit(0)
else:
    print("⚠️  Some validation checks failed.")
    print("Review the issues above and fix before deployment.\n")
    sys.exit(1)
