# Setup options
name = "AFToolKit"  # This is the name on PyPI/pip list
author = "Shashkova Tatiana, Sindeeva Maria, Ivanisenko Nikita, Telepov Alexander"
version = "1.0.0"
description = "Python library for routine protein engineering tasks using AlphaFold2."

# Ensure these keys match the folder name exactly
package_dir = {"AFToolKit": "AFToolKit"}

# find_packages will now look for the AFToolKit folder
packages = find_packages(exclude=['tests'])

package_data = {
    "AFToolKit": [
        "processing/*",
        "processing/openfold/*",
        "models/*"
    ]
}

entry_points = {
    "console_scripts": [
        "run_protein_task=AFToolKit.processing.run_protein_task:main",
        "run_protein_complex_task=AFToolKit.processing.run_protein_complex_task:main"
    ],
}
