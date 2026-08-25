from huggingface_hub import HfApi

repo_id = "mistralai/Mistral-Small-4-119B-2603"
info = HfApi().model_info(repo_id)

print("repo_id :", repo_id)
print("revision:", info.sha)
