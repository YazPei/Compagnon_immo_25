from setuptools import setup, find_packages

setup(
    name="compagnon_immo",
    version="1.0.0",
    description="Prédiction des prix immobiliers par séries temporelles",
    author="Pedro Ketsia",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.10.0",
        "scikit-learn>=1.1.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pyyaml>=6.0",
        "joblib>=1.1.0"
    ],
    entry_points={
        'console_scripts': [
            'compagnon-immo=main:main',
        ],
    },
)
