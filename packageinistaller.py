import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def installpackages():
    with open("requirements.txt", "r") as f:
        packages = f.readlines()

    for package in packages:
        package = package.strip()
        if package:
            try:
                __import__(package.split('==')[0])
            except ImportError:
                print(f"Installing {package}...")
                install(package)
            else:
                print(f"{package} is already installed.")