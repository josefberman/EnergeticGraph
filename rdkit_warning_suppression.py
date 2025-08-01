# Add this to the top of your Python scripts to suppress RDKit warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")

# Or add this to your .env file:
# PYTHONWARNINGS=ignore::RuntimeWarning:importlib._bootstrap:*
