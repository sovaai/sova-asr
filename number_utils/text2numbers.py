from number_utils.russian_numbers import RussianNumbers
import numpy as np


class TextToNumbers:
    def __init__(self):
        self.russian_numbers = RussianNumbers()

    def convert(self, text_line):
        if not text_line:
            return text_line

        parsed_list, result_text_list = self.russian_numbers.parse(text=text_line)
        converted_text = ""
        parsed_idx = 0
        result_text_list_len = len(result_text_list)

        for i, element in enumerate(result_text_list):
            if element == "":
                converted_text += str(parsed_list[parsed_idx].value)
                parsed_idx += 1

            else:
                converted_text += element

            if i < result_text_list_len - 1:
                converted_text += " "

        converted_text = self.float_postprocessing(converted_text)

        return converted_text

    @staticmethod
    def float_postprocessing(converted_text):
        if "минус" in converted_text and " и " not in converted_text:
            for x in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                if x in converted_text:
                    converted_text = converted_text.replace("минус ", "-")

        if "точка" in converted_text:
            try:
                for idx, val in enumerate(converted_text):
                    if val == "т":
                        if converted_text[idx + 4] == "а" \
                                and converted_text[idx - 2] in ["0", "1", "2", "3", "4",
                                                                "5", "6", "7", "8", "9"] \
                                and converted_text[idx + 6] in ["0", "1", "2", "3", "4",
                                                                "5", "6", "7", "8", "9"] \
                                and "0." in converted_text[idx + 6:]:
                            converted_text = converted_text.replace(" точка ", ".").replace("0.", "")
                        if converted_text[idx + 4] == "а" \
                                and converted_text[idx - 2] in ["0", "1", "2", "3", "4",
                                                                "5", "6", "7", "8", "9"] \
                                and converted_text[idx + 6] in ["0", "1", "2", "3", "4",
                                                                "5", "6", "7", "8", "9"] \
                                and "0." not in converted_text[idx + 6:]:
                            converted_text = converted_text.replace(" точка ", ".")

            except IndexError:
                pass

        if "запятая" in converted_text:
            try:
                for idx, val in enumerate(converted_text):
                    if val == "з":
                        if converted_text[idx + 6] == "я" \
                                and converted_text[idx - 2] in ["0", "1", "2", "3", "4",
                                                                "5", "6", "7", "8", "9"] \
                                and converted_text[idx + 8] in ["0", "1", "2", "3", "4",
                                                                "5", "6", "7", "8", "9"] \
                                and "0." in converted_text[idx + 8:]:
                            converted_text = converted_text.replace(" запятая ", ".").replace("0.", "")
                        if converted_text[idx + 6] == "я" \
                                and converted_text[idx - 2] in ["0", "1", "2", "3", "4",
                                                                "5", "6", "7", "8", "9"] \
                                and converted_text[idx + 8] in ["0", "1", "2", "3", "4",
                                                                "5", "6", "7", "8", "9"] \
                                and "0." not in converted_text[idx + 8:]:
                            converted_text = converted_text.replace(" запятая ", ".")

            except IndexError:
                pass

        if " и " in converted_text:
            try:
                for idx, val in enumerate(converted_text):
                    if val == "и" \
                            and converted_text[idx + 2] in ["0", "1", "2", "3", "4",
                                                            "5", "6", "7", "8", "9"] \
                            and converted_text[idx - 1] == " " \
                            and converted_text[idx - 2] in ["0", "1", "2", "3", "4",
                                                            "5", "6", "7", "8", "9"]:

                        if " 0." in converted_text:
                            converted_text = converted_text.replace("0.", ".").replace(" и ", "")

                            if "после запятой" in converted_text:
                                converted_text = converted_text.replace(" после запятой", "")

                        if "после запятой" in converted_text:
                            converted_text = converted_text.replace(" и ", ".").replace(" после запятой", "")

                        else:
                            converted_text = converted_text.replace(" и ", ".")
            except IndexError:
                return converted_text

            try:
                converted_text = str(np.round(np.float32(converted_text), 5))
            except (ValueError, TypeError):
                pass

        return converted_text


if __name__ == "__main__":
    text2numbers = TextToNumbers()

    while True:
        text_line = input("Введите ваш текст:\n")
        converted_line = text2numbers.convert(text_line)
        print(f"\nРаспознанное: {converted_line}\n\n")
