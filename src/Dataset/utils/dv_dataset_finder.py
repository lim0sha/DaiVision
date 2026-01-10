from pathlib import Path


def find_dv_dataset(datasets_dir: Path) -> Path:
    if not datasets_dir.exists():
        raise FileNotFoundError("[ERROR]: папка datasets не найдена")

    candidates = [
        d for d in datasets_dir.iterdir()
        if d.is_dir() and d.name.startswith("ChatExport")
    ]

    if not candidates:
        raise FileNotFoundError(
            "[ERROR]: папка ChatExport* не найдена в datasets"
        )

    if len(candidates) > 1:
        raise RuntimeError(
            f"[ERROR]: найдено несколько ChatExport*: {[d.name for d in candidates]}"
        )

    return candidates[0]
