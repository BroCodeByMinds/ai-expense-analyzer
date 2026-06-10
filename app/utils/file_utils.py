from pathlib import Path


def get_file_extension(file_path: str) -> str:

    return Path(file_path).suffix.replace(".", "").lower()