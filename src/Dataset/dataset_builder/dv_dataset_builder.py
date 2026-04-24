import json

import pandas as pd

from src.Сonfigs.common_paths import DV_RAW_CSV


class DatasetBuilder:
    def __init__(self, path_to_json: str):
        self.jsonFd = open(path_to_json, "r", encoding="utf-8")
        self.data = json.load(self.jsonFd)

        self.bot_id = 0
        self.user_id = 0

        for message in self.data["messages"]:
            if message.get("from") == "Дайвинчик | Leo – знакомства, общение и новые друзья":
                self.bot_id = message.get("from_id")
            else:
                self.user_id = message.get("from_id")

            if self.bot_id != 0 and self.user_id != 0:
                break

        self.parse_exception = ["🚀 Смотреть анкеты", "Нет", "1 🚀", "1 👍"]

    def build_dataset(self) -> pd.DataFrame:
        rows = []

        temp_files = []
        profile_id = 0

        for message in self.data["messages"]:
            from_id = message.get("from_id")
            if from_id == self.bot_id:
                if "photo" in message:
                    temp_files.append(message["photo"])
                elif "file" in message:
                    temp_files.append(message["file"])
            elif from_id == self.user_id:
                text = message.get("text")

                if not text or text in self.parse_exception:
                    continue
                profile_liked = 0 if text == "👎" else 1

                for idx, file_path in enumerate(temp_files):
                    rows.append({
                        "profile_id": profile_id,
                        "image_path": file_path,
                        "image_index": idx,
                        "profile_liked": profile_liked,
                    })

                if temp_files:
                    profile_id += 1
                temp_files = []

        return pd.DataFrame(
            rows,
            columns=[
                "profile_id",
                "image_path",
                "image_index",
                "profile_liked",
            ],
        )

    def export_to_csv(self, output_path=DV_RAW_CSV):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.build_dataset()
        df.to_csv(output_path, index=False, encoding="utf-8")

    def __del__(self):
        self.jsonFd.close()
