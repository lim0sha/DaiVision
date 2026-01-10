"""
Модуль для поиска директории датасета DaiVision в указанной директории.

Этот модуль предоставляет функцию для поиска единственной директории с префиксом
"ChatExport" внутри указанной директории datasets. Функция используется для
нахождения экспортированного чата с данными из приложения "Дайвинчик".
"""

from pathlib import Path


def find_dv_dataset(datasets_dir: Path) -> Path:
    """
    Ищет директорию с датасетом DaiVision в указанной директории.

    Функция ищет подкаталог с именем, начинающимся на "ChatExport",
    внутри указанной директории datasets, и возвращает путь к этой директории.
    Если таких директорий не найдено или их больше одной, выбрасывается
    соответствующее исключение.

    Args:
        datasets_dir (Path): Директория, в которой нужно искать подкаталог ChatExport*

    Returns:
        Path: Путь к найденной директории с префиксом "ChatExport"

    Raises:
        FileNotFoundError:
            - если директория datasets_dir не существует
            - если не найдены подкаталоги с префиксом "ChatExport"
        RuntimeError:
            - если найдено более одной директории с префиксом "ChatExport"
    """
    # Проверяем существование директории datasets
    if not datasets_dir.exists():
        raise FileNotFoundError("[ERROR]: папка datasets не найдена")

    # Находим все подкаталоги, начинающиеся с "ChatExport"
    candidates = [
        d for d in datasets_dir.iterdir()
        if d.is_dir() and d.name.startswith("ChatExport")
    ]

    # Если не найдено ни одного подходящего каталога
    if not candidates:
        raise FileNotFoundError(
            "[ERROR]: папка ChatExport* не найдена в datasets"
        )

    # Если найдено более одного каталога
    if len(candidates) > 1:
        raise RuntimeError(
            f"[ERROR]: найдено несколько ChatExport*: {[d.name for d in candidates]}"
        )

    # Возвращаем единственный найденный каталог
    return candidates[0]
