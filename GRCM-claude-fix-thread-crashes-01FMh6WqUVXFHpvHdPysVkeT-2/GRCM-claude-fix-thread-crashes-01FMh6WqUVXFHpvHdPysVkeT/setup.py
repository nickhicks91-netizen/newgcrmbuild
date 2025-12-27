"""
Setup script for GRCM - Grounded Resonant Consciousness Module
Production-ready implementation with comprehensive extras for optimization, logging, serving, and UI.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ""

# Version
VERSION = "1.0.0"

# Core requirements
INSTALL_REQUIRES = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "pyyaml>=6.0",
]

# Optional dependencies
EXTRAS_REQUIRE = {
    "optimization": [
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
    ],
    "logging": [
        "mlflow>=2.8.0",
    ],
    "ui": [
        "gradio>=3.50.0",
    ],
    "serving": [
        "bentoml>=1.1.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "prometheus-client>=0.18.0",
        "redis>=5.0.0",
    ],
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-timeout>=2.1.0",
        "pytest-benchmark>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.1.0",
        "mypy>=1.5.0",
    ],
    "docs": [
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "sphinx-autodoc-typehints>=1.22.0",
        "myst-parser>=1.0.0",
    ],
}

# Convenience "all" extra
EXTRAS_REQUIRE["all"] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

setup(
    name="grcm",
    version=VERSION,
    author="GRCM Contributors",
    author_email="",
    description="Grounded Resonant Consciousness Module - Production-ready consciousness simulation with resonant attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nickhicks91-netizen/newdew",
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'examples.*', 'docs', 'docs.*', 'deployment', 'deployment.*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "grcm=grcm.__main__:main",
        ],
    },
    include_package_data=True,
    package_data={
        "grcm": ["py.typed", "*.yaml", "*.yml"],
        "": ["config/*.yaml", "config/*.yml"],
    },
    zip_safe=False,
    keywords=[
        "consciousness",
        "artificial-intelligence",
        "integrated-information-theory",
        "resonance",
        "attention",
        "pytorch",
        "machine-learning",
        "cognitive-science",
        "phi",
        "qualia",
    ],
    project_urls={
        "Documentation": "https://grcm.readthedocs.io",
        "Source": "https://github.com/nickhicks91-netizen/newdew",
        "Bug Reports": "https://github.com/nickhicks91-netizen/newdew/issues",
        "Discussions": "https://github.com/nickhicks91-netizen/newdew/discussions",
    },
)
