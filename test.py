from datasets import load_dataset
import json






from huggingface_hub import list_repo_files
files = list_repo_files("jhu-clsp/SARA", repo_type="dataset")
for f in files:
    print(f)