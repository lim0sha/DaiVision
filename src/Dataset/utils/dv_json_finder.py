"""
Модуль для поиска файла result.json в директориях экспорта чата Telegram.

Этот модуль предоставляет функцию для поиска единственного файла result.json
в подкаталогах с префиксом "ChatExport" внутри указанной директории datasets.
"""

from pathlib import Path


def find_result_json(datasets_dir):
    """
    Ищет файл result.json в подкаталогах с префиксом "ChatExport".

    Функция ищет в подкаталогах с именем, начинающимся на "ChatExport",
    внутри указанной директории datasets, и возвращает путь к первому найденному
    файлу result.json. Если файлов больше одного или не найдено ни одного,
    выбрасывается соответствующее исключение.

    Args:
        datasets_dir (Path): Директория, в которой нужно искать подкаталоги ChatExport*

    Returns:
        Path: Путь к найденному файлу result.json

    Raises:
        FileNotFoundError:
            - если директория datasets_dir не существует
            - если не найдены подкаталоги с префиксом "ChatExport"
            - если файл result.json не найден в найденных подкаталогах
        Exception:
            - если найдено более одного файла result.json
    """
    # Проверяем существование директории datasets
    if not datasets_dir.exists():
        raise FileNotFoundError("[ERROR]: папка datasets не найдена")

    # Находим все подкаталоги, начинающиеся с "ChatExport"
    chat_export_dirs = [
        d for d in datasets_dir.iterdir()
        if d.is_dir() and d.name.startswith("ChatExport")
    ]

    # Если не найдено ни одного подходящего каталога
    if not chat_export_dirs:
        raise FileNotFoundError("[ERROR]: папки ChatExport* не найдены в datasets")

    # Поиск файлов result.json во всех найденных каталогах
    found_files = []

    for chat_dir in chat_export_dirs:
        found_files.extend(chat_dir.rglob("result.json"))

    # Если файлов не найдено
    if not found_files:
        raise FileNotFoundError(
            "[ERROR]: result.json не найден в папках ChatExport*"
        )

    # Если найдено более одного файла
    if len(found_files) > 1:
        raise Exception(
            "[ERROR]: найдено несколько result.json в папках ChatExport*"
        )

    # Возвращаем единственный найденный файл
    return found_files[0]

