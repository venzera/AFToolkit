from setuptools import setup, find_packages

setup(
    name="AFToolKit",
    version="1.0.0",
    author="Shashkova Tatiana, Sindeeva Maria, Ivanisenko Nikita, Telepov Alexander",
    description="Python library for routine protein engineering tasks using AlphaFold2.",
    packages=find_packages(),
    package_data={
        "AFToolKit": [
            "processing/*",
            "processing/openfold/*",
            "models/*",
        ]
    },
    entry_points={
        "console_scripts": [
            "run_protein_task=AFToolKit.processing.run_protein_task:main",
            "run_protein_complex_task=AFToolKit.processing.run_protein_complex_task:main",
        ],
    },
)
