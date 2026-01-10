import json
import pandas as pd


class DatasetBuilder:
    def __init__(self, path_to_json):
        with open(path_to_json, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.bot_name = "Дайвинчик | Leo – знакомства, общение и новые друзья"
        self.parse_exception = {"🚀 Смотреть анкеты", "Нет", "1 🚀", "1 👍"}

    def build_dataset(self):
        rows = []

        current_profile_photos = []
        profile_id = 0

        for msg in self.data["messages"]:
            if msg.get("from") == self.bot_name:
                if "photo" in msg:
                    current_profile_photos.append(msg["photo"])
                elif "file" in msg:
                    current_profile_photos.append(msg["file"])

            else:
                text = msg.get("text")

                if text in self.parse_exception:
                    continue

                if text not in {"❤️", "👎"}:
                    continue

                profile_liked = 1 if text == "❤️" else 0

                if profile_liked == 0:
                    final_label = 0
                else:
                    final_label = 1

                for idx, photo_path in enumerate(current_profile_photos):
                    rows.append({
                        "profile_id": profile_id,
                        "image_path": photo_path,
                        "image_index": idx,
                        "profile_liked": final_label
                    })

                profile_id += 1
                current_profile_photos = []

        return pd.DataFrame(
            rows,
            columns=["profile_id", "image_path", "image_index", "profile_liked"]
        )

    def export_to_csv(self, output_path="files/processed/dv_dataset_raw.csv"):
        df = self.build_dataset()
        df.to_csv(output_path, index=False, encoding="utf-8")
