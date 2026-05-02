from setuptools import setup, find_packages

setup(
    name="traffic-forecasting-system",
    version="1.0.0",
    description="GPU-Accelerated GraphCast-Style Spatiotemporal Traffic Forecasting System",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.2.0",
    ],
    python_requires=">=3.8",
)
