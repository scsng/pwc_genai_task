import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def setup_logging(log_level: str = None):
    """
    Set up logging configuration to output to console.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                  If None, will use LOG_LEVEL from environment or default to INFO.
    """
    # Get log level from parameter, env var, or default to INFO
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    
    # Prevent duplicate logs from other libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('streamlit').setLevel(logging.WARNING)
