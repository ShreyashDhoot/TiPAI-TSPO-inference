from huggingface_hub import HfApi

# Your token from https://huggingface.co/settings/tokens
MY_TOKEN = "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"

api = HfApi(token=MY_TOKEN)

api.upload_folder(
    folder_path="results",
    repo_id="ShreyashDhoot/sneak-prompt-baseline",
    repo_type="dataset"
)
