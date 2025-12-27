"""
GRCM Structure Verification Script
Checks that all required files and modules are present
"""
import sys
from pathlib import Path

def check_file_exists(path: str) -> bool:
    """Check if a file exists"""
    exists = Path(path).exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {path}")
    return exists

def main():
    print("=" * 60)
    print("GRCM Structure Verification")
    print("=" * 60)

    all_checks_passed = True

    # Check package structure
    print("\n[1] Core Package Files:")
    files = [
        "grcm/__init__.py",
        "grcm/config.py",
        "grcm/core.py",
        "grcm/trainer.py",
    ]
    all_checks_passed &= all(check_file_exists(f) for f in files)

    # Check modules
    print("\n[2] Module Files:")
    modules = [
        "grcm/modules/__init__.py",
        "grcm/modules/grounding.py",
        "grcm/modules/embedding.py",
        "grcm/modules/attention.py",
        "grcm/modules/desire.py",
        "grcm/modules/memory.py",
        "grcm/modules/reflection.py",
        "grcm/modules/qualia.py",
        "grcm/modules/threading.py",
        "grcm/modules/phi.py",
        "grcm/modules/body.py",
    ]
    all_checks_passed &= all(check_file_exists(f) for f in modules)

    # Check configuration
    print("\n[3] Configuration Files:")
    configs = [
        "config/grcm_default.yaml",
    ]
    all_checks_passed &= all(check_file_exists(f) for f in configs)

    # Check documentation
    print("\n[4] Documentation Files:")
    docs = [
        "README_GRCM.md",
        "docs/ARCHITECTURE.md",
        "docs/PHASE1_SUMMARY.md",
    ]
    all_checks_passed &= all(check_file_exists(f) for f in docs)

    # Check examples
    print("\n[5] Example Files:")
    examples = [
        "examples/basic_usage.py",
    ]
    all_checks_passed &= all(check_file_exists(f) for f in examples)

    # Check setup files
    print("\n[6] Setup Files:")
    setup_files = [
        "setup.py",
        "requirements_grcm.txt",
    ]
    all_checks_passed &= all(check_file_exists(f) for f in setup_files)

    # Summary
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✓ All structure checks passed!")
        print("=" * 60)
        print("\nPhase 1 Complete:")
        print("  - 13 module files")
        print("  - 1 configuration file")
        print("  - 3 documentation files")
        print("  - 1 example file")
        print("  - 2 setup files")
        print("\nReady for Phase 2: Optimization & Export")
        return 0
    else:
        print("✗ Some files are missing!")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
