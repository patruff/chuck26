import os
# Set UTF-8 as default encoding for Windows
os.environ["PYTHONIOENCODING"] = "utf-8"

from huggingface_hub import login
from word_tool import WordPhoneTool

# Login to Hugging Face
login("hf_tokenblahblabhalbh")

# Create and push tool
tool = WordPhoneTool()
try:
    tool.push_to_hub(
        repo_id="patruff/word-phone",
        token="hf_tokenblahblahblah",
        private=False
    )
    print("Successfully pushed tool to hub!")
except Exception as e:
    print(f"Error pushing to hub: {e}")
