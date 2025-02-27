"""
Interpretability experiments package.
"""
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'experiment.log'), mode='a')
    ]
)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)