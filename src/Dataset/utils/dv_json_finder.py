def find_result_json(datasets_dir):
    if not datasets_dir.exists():
        raise FileNotFoundError("[ERROR]: папка datasets не найдена")

    chat_export_dirs = [
        d for d in datasets_dir.iterdir()
        if d.is_dir() and d.name.startswith("ChatExport")
    ]

    if not chat_export_dirs:
        raise FileNotFoundError("[ERROR]: папки ChatExport* не найдены в datasets")

    found_files = []

    for chat_dir in chat_export_dirs:
        found_files.extend(chat_dir.rglob("result.json"))

    if not found_files:
        raise FileNotFoundError(
            "[ERROR]: result.json не найден в папках ChatExport*"
        )

    if len(found_files) > 1:
        raise Exception(
            "[ERROR]: найдено несколько result.json в папках ChatExport*"
        )

    return found_files[0]
