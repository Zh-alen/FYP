import importlib

required_packages = [
    'jax', 'jaxlib', 'openpyxl', 'docx', 
    'pandas', 'numpy', 'matplotlib', 'nbformat'
]

print("Checking environment configuration...")
for package in required_packages:
    try:
        importlib.import_module(package)
        print(f"OK: {package}")
    except ImportError:
        print(f"MISSING: {package}")

print("Environment check completed.")