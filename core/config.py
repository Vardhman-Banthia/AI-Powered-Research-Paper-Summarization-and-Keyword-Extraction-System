import os
import sys
from dotenv import load_dotenv

# Automatically look for a .env file in the project root
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def validate_config():
    """
    Validates backend configuration. Logs developer errors securely.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY.strip() == "" or GEMINI_API_KEY == "your_actual_api_key_here":
        print(
            "ERROR:\n"
            "Gemini API key not found or is invalid.\n"
            "Please configure GEMINI_API_KEY inside the .env file or platform variables.",
            file=sys.stderr
        )
        return False
    return True
