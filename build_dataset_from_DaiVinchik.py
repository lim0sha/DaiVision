import json
import csv


class BuildDatasetFromDV:
    def __init__(self, path_to_json):
        self.jsonFd = open(path_to_json, "r", encoding="utf-8")
        self.data = json.load(self.jsonFd)

        self.botId, self.userId = 0, 0
        for message in self.data["messages"]:
            if message["from"] == "Дайвинчик | Leo – знакомства, общение и новые друзья":
                self.botId = message["from_id"]
            else:
                self.userId = message["from_id"]

            if self.botId != 0 and self.userId != 0:
                break

        self.parseException = ["🚀 Смотреть анкеты", "Нет", "1 🚀", "1 👍"]

    def parse_to_dict(self):
        filesAndRatings = {"0": [],
                           "1": []}
        tempArray = []

        for message in self.data["messages"]:
            if message["from_id"] == self.botId:
                if "photo" in message:
                    tempArray.append(message["photo"])
                elif "file" in message:
                    tempArray.append(message["file"])
            elif message["from_id"] == self.userId:
                if "text" in message and message["text"] not in self.parseException:
                    filesAndRatings["0" if message["text"] == "👎" else "1"].extend(tempArray)
                    tempArray = []

        return filesAndRatings

    def export_to_csv(self):
        filesAndRatings = self.parse_to_dict()

        csvFd = open("datasets/dataset_from_DV.csv", "w", newline="", encoding="utf-8")
        csvWriter = csv.writer(csvFd)

        for key, values in filesAndRatings.items():
            for value in values:
                csvWriter.writerow([key, value])

        csvFd.close()

    def __del__(self):
        self.jsonFd.close()


if __name__ == '__main__':
    buildDatasetFromDV = BuildDatasetFromDV(
        "C:/Users/makarkme/PycharmProjects/Dai-CVinchik/datasets/ChatExport_2026-01-09/result.json")
    buildDatasetFromDV.export_to_csv()
