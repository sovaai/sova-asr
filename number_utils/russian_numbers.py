import string, re
import numpy as np


class Numeral:
    def __init__(self, value, level, is_multiplier, is_eleven_to_nineteen=False):
        self.value = value
        self.level = level
        self.is_multiplier = is_multiplier
        self.is_eleven_to_nineteen = is_eleven_to_nineteen


class NumericToken:
    def __init__(self, numeral, error=0):
        self.numeral = numeral
        self.error = error
        self.is_significant = False


class ParserResult:
    def __init__(self, value, error=0):
        self.value = value
        self.error = error


class RussianNumbers:
    def __init__(self):
        self.tokens_fractions = {
            "целых": Numeral(1, 0, True),
            "целым": Numeral(1, 0, True),
            "целой": Numeral(1, 0, True),
            "целая": Numeral(1, 0, True),

            "точка": Numeral(0.1, 0, True),
            "запятая": Numeral(1, 0, True),

            "десятых": Numeral(0.1, -1, True),
            "десятым": Numeral(0.1, -1, True),
            "десятая": Numeral(0.1, -1, True),
            "сотых": Numeral(0.01, -3, True),
            "сотым": Numeral(0.01, -3, True),
            "сотая": Numeral(0.01, -3, True),
            "тысячных": Numeral(0.001, -4, True),
            "тысячным": Numeral(0.001, -4, True),
            "тысячная": Numeral(0.001, -4, True),
            "десятитысячных": Numeral(0.0001, -5, True),
            "десятитысячная": Numeral(0.0001, -5, True),

            # "половиной": Numeral(0.5, -1, False),
        }

        self.tokens = {
            "ноль": Numeral(0, 1, False),
            "нулю": Numeral(0, 1, False),
            "нолю": Numeral(0, 1, False),
            "ноля": Numeral(0, 1, False),

            "полтора": Numeral(1.5, 1, False),
            "полторы": Numeral(1.5, 1, False),

            "один": Numeral(1, 1, False),
            "одна": Numeral(1, 1, False),
            "одной": Numeral(1, 1, False),
            "первое": Numeral(1, 1, False),
            "первый": Numeral(1, 1, False),
            "первая": Numeral(1, 1, False),
            "первого": Numeral(1, 1, False),
            "первой": Numeral(1, 1, False),
            "первом": Numeral(1, 1, False),

            "два": Numeral(2, 1, False),
            "две": Numeral(2, 1, False),
            "двум": Numeral(2, 1, False),
            "двух": Numeral(2, 1, False),
            "второе": Numeral(2, 1, False),
            "второй": Numeral(2, 1, False),
            "вторая": Numeral(2, 1, False),
            "второго": Numeral(2, 1, False),
            "втором": Numeral(2, 1, False),

            "три": Numeral(3, 1, False),
            "трем": Numeral(3, 1, False),
            "трех": Numeral(3, 1, False),
            "трёх": Numeral(3, 1, False),
            "трём": Numeral(3, 1, False),
            "третье": Numeral(3, 1, False),
            "третий": Numeral(3, 1, False),
            "третья": Numeral(3, 1, False),
            "третьего": Numeral(3, 1, False),
            "третьей": Numeral(3, 1, False),
            "третьем": Numeral(3, 1, False),

            "четыре": Numeral(4, 1, False),
            "четырем": Numeral(4, 1, False),
            "четырех": Numeral(4, 1, False),
            "четырём": Numeral(4, 1, False),
            "четырёх": Numeral(4, 1, False),
            "четвертое": Numeral(4, 1, False),
            "четвертый": Numeral(4, 1, False),
            "четвертая": Numeral(4, 1, False),
            "четвертого": Numeral(4, 1, False),
            "четвертой": Numeral(4, 1, False),
            "четвертом": Numeral(4, 1, False),

            "пять": Numeral(5, 1, False),
            "пяти": Numeral(5, 1, False),
            "пятое": Numeral(5, 1, False),
            "пятый": Numeral(5, 1, False),
            "пятая": Numeral(5, 1, False),
            "пятого": Numeral(5, 1, False),
            "пятой": Numeral(5, 1, False),
            "пятом": Numeral(5, 1, False),

            "шесть": Numeral(6, 1, False),
            "шести": Numeral(6, 1, False),
            "шестое": Numeral(6, 1, False),
            "шестой": Numeral(6, 1, False),
            "шестая": Numeral(6, 1, False),
            "шестого": Numeral(6, 1, False),
            "шестом": Numeral(6, 1, False),

            "семь": Numeral(7, 1, False),
            "семи": Numeral(7, 1, False),
            "седьмое": Numeral(7, 1, False),
            "седьмой": Numeral(7, 1, False),
            "седьмая": Numeral(7, 1, False),
            "седьмого": Numeral(7, 1, False),
            "седьмом": Numeral(7, 1, False),

            "восемь": Numeral(8, 1, False),
            "восьми": Numeral(8, 1, False),
            "восьмое": Numeral(8, 1, False),
            "восьмой": Numeral(8, 1, False),
            "восьмая": Numeral(8, 1, False),
            "восьмого": Numeral(8, 1, False),
            "восьмом": Numeral(8, 1, False),

            "девять": Numeral(9, 1, False),
            "девяти": Numeral(9, 1, False),
            "девятое": Numeral(9, 1, False),
            "девятый": Numeral(9, 1, False),
            "девятая": Numeral(9, 1, False),
            "девятого": Numeral(9, 1, False),
            "девятой": Numeral(9, 1, False),
            "девятом": Numeral(9, 1, False),

            "десять": Numeral(10, 1, False),
            "десятью": Numeral(10, 1, False),
            "десяти": Numeral(10, 1, False),
            "десятое": Numeral(10, 1, False),
            "десятый": Numeral(10, 1, False),
            "десятая": Numeral(10, 1, False),
            "десятой": Numeral(10, 1, False),
            "десятого": Numeral(10, 1, False),
            "десятом": Numeral(10, 1, False),

            "одиннадцать": Numeral(11, 1, False, True),
            "одиннадцатью": Numeral(11, 1, False, True),
            "одиннадцати": Numeral(11, 1, False, True),
            "одиннадцатое": Numeral(11, 1, False, True),
            "одиннадцатый": Numeral(11, 1, False, True),
            "одиннадцатая": Numeral(11, 1, False, True),
            "одиннадцатого": Numeral(11, 1, False, True),
            "одиннадцатой": Numeral(11, 1, False, True),
            "одиннадцатом": Numeral(11, 1, False, True),

            "двенадцать": Numeral(12, 1, False, True),
            "двенадцатью": Numeral(12, 1, False, True),
            "двенадцати": Numeral(12, 1, False, True),
            "двенадцатое": Numeral(12, 1, False, True),
            "двенадцатый": Numeral(12, 1, False, True),
            "двенадцатая": Numeral(12, 1, False, True),
            "двенадцатого": Numeral(12, 1, False, True),
            "двенадцатой": Numeral(12, 1, False, True),
            "двенадцатом": Numeral(12, 1, False, True),

            "тринадцать": Numeral(13, 1, False, True),
            "тринадцатью": Numeral(13, 1, False, True),
            "тринадцати": Numeral(13, 1, False, True),
            "тринадцатое": Numeral(13, 1, False, True),
            "тринадцатый": Numeral(13, 1, False, True),
            "тринадцатая": Numeral(13, 1, False, True),
            "тринадцатого": Numeral(13, 1, False, True),
            "тринадцатой": Numeral(13, 1, False, True),
            "тринадцатом": Numeral(13, 1, False, True),

            "четырнадцать": Numeral(14, 1, False, True),
            "четырнадцатью": Numeral(14, 1, False, True),
            "четырнадцати": Numeral(14, 1, False, True),
            "четырнадцатое": Numeral(14, 1, False, True),
            "четырнадцатый": Numeral(14, 1, False, True),
            "четырнадцатая": Numeral(14, 1, False, True),
            "четырнадцатого": Numeral(14, 1, False, True),
            "четырнадцатой": Numeral(14, 1, False, True),
            "четырнадцатом": Numeral(14, 1, False, True),

            "пятнадцать": Numeral(15, 1, False, True),
            "пятнадцатью": Numeral(15, 1, False, True),
            "пятнадцати": Numeral(15, 1, False, True),
            "пятнадцатое": Numeral(15, 1, False, True),
            "пятнадцатый": Numeral(15, 1, False, True),
            "пятнадцатая": Numeral(15, 1, False, True),
            "пятнадцатого": Numeral(15, 1, False, True),
            "пятнадцатой": Numeral(15, 1, False, True),
            "пятнадцатом": Numeral(15, 1, False, True),

            "шестнадцать": Numeral(16, 1, False, True),
            "шестнадцатью": Numeral(16, 1, False, True),
            "шестнадцати": Numeral(16, 1, False, True),
            "шестнадцатое": Numeral(16, 1, False, True),
            "шестнадцатый": Numeral(16, 1, False, True),
            "шестнадцатая": Numeral(16, 1, False, True),
            "шестнадцатого": Numeral(16, 1, False, True),
            "шестнадцатой": Numeral(16, 1, False, True),
            "шестнадцатом": Numeral(16, 1, False, True),

            "семнадцать": Numeral(17, 1, False, True),
            "семнадцатью": Numeral(17, 1, False, True),
            "семнадцати": Numeral(17, 1, False, True),
            "семнадцатое": Numeral(17, 1, False, True),
            "семнадцатый": Numeral(17, 1, False, True),
            "семнадцатая": Numeral(17, 1, False, True),
            "семнадцатого": Numeral(17, 1, False, True),
            "семнадцатой": Numeral(17, 1, False, True),
            "семнадцатом": Numeral(17, 1, False, True),

            "восемнадцать": Numeral(18, 1, False, True),
            "восемнадцатью": Numeral(18, 1, False, True),
            "восемнадцати": Numeral(18, 1, False, True),
            "восемнадцатое": Numeral(18, 1, False, True),
            "восемнадцатый": Numeral(18, 1, False, True),
            "восемнадцатая": Numeral(18, 1, False, True),
            "восемнадцатого": Numeral(18, 1, False, True),
            "восемнадцатой": Numeral(18, 1, False, True),
            "восемнадцатом": Numeral(18, 1, False, True),

            "девятнадцать": Numeral(19, 1, False, True),
            "девятнадцатью": Numeral(19, 1, False, True),
            "девятнадцати": Numeral(19, 1, False, True),
            "девятнадцатое": Numeral(19, 1, False, True),
            "девятнадцатый": Numeral(19, 1, False, True),
            "девятнадцатая": Numeral(19, 1, False, True),
            "девятнадцатого": Numeral(19, 1, False, True),
            "девятнадцатой": Numeral(19, 1, False, True),
            "девятнадцатом": Numeral(19, 1, False, True),

            "двадцать": Numeral(20, 2, False),
            "двадцатью": Numeral(20, 2, False),
            "двадцати": Numeral(20, 2, False),
            "двадцатое": Numeral(20, 2, False),
            "двадцатый": Numeral(20, 2, False),
            "двадцатая": Numeral(20, 2, False),
            "двадцатого": Numeral(20, 2, False),
            "двадцатой": Numeral(20, 2, False),
            "двадцатом": Numeral(20, 2, False),

            "тридцать": Numeral(30, 2, False),
            "тридцатью": Numeral(30, 2, False),
            "тридцати": Numeral(30, 2, False),
            "тридцатое": Numeral(30, 2, False),
            "тридцатый": Numeral(30, 2, False),
            "тридцатая": Numeral(30, 2, False),
            "тридцатого": Numeral(30, 2, False),
            "тридцатой": Numeral(30, 2, False),
            "тридцатом": Numeral(30, 2, False),

            "сорок": Numeral(40, 2, False),
            "сорока": Numeral(40, 2, False),
            "сороковое": Numeral(40, 2, False),
            "сороковой": Numeral(40, 2, False),
            "сороковая": Numeral(40, 2, False),
            "сорокового": Numeral(40, 2, False),
            "сороковом": Numeral(40, 2, False),

            "пятьдесят": Numeral(50, 2, False),
            "пятидесяти": Numeral(50, 2, False),
            "пятьюдесятью": Numeral(50, 2, False),
            "пятидесятое": Numeral(50, 2, False),
            "пятидесятый": Numeral(50, 2, False),
            "пятидесятая": Numeral(50, 2, False),
            "пятидесятого": Numeral(50, 2, False),
            "пятидесятой": Numeral(50, 2, False),
            "пятидесятом": Numeral(50, 2, False),

            "шестьдесят": Numeral(60, 2, False),
            "шестидесяти": Numeral(60, 2, False),
            "шестьюдесятью": Numeral(60, 2, False),
            "шестидесятое": Numeral(60, 2, False),
            "шестидесятый": Numeral(60, 2, False),
            "шестидесятая": Numeral(60, 2, False),
            "шестидесятого": Numeral(60, 2, False),
            "шестидесятой": Numeral(60, 2, False),
            "шестидесятом": Numeral(60, 2, False),

            "семьдесят": Numeral(70, 2, False),
            "семидесяти": Numeral(70, 2, False),
            "семьюдесятью": Numeral(70, 2, False),
            "семидесятое": Numeral(70, 2, False),
            "семидесятый": Numeral(70, 2, False),
            "семидесятая": Numeral(70, 2, False),
            "семидесятого": Numeral(70, 2, False),
            "семидесятой": Numeral(70, 2, False),
            "семидесятом": Numeral(70, 2, False),

            "восемьдесят": Numeral(80, 2, False),
            "восьмидесяти": Numeral(80, 2, False),
            "восемьюдесятью": Numeral(80, 2, False),
            "восьмидесятое": Numeral(80, 2, False),
            "восьмидесятый": Numeral(80, 2, False),
            "восьмидесятая": Numeral(80, 2, False),
            "восьмидесятого": Numeral(80, 2, False),
            "восьмидесятой": Numeral(80, 2, False),
            "восьмидесятом": Numeral(80, 2, False),

            "девяносто": Numeral(90, 2, False),
            "девяноста": Numeral(90, 2, False),
            "девяностое": Numeral(90, 2, False),
            "девяностый": Numeral(90, 2, False),
            "девяностая": Numeral(90, 2, False),
            "девяностого": Numeral(90, 2, False),
            "девяностой": Numeral(90, 2, False),
            "девяностом": Numeral(90, 2, False),

            "сто": Numeral(100, 3, False),
            "ста": Numeral(100, 3, False),
            "сотое": Numeral(100, 3, False),
            "сотый": Numeral(100, 3, False),
            # "сотая": Numeral(100, 3, False),
            "сотого": Numeral(100, 3, False),
            "сотой": Numeral(100, 3, False),
            "сотом": Numeral(100, 3, False),

            "двести": Numeral(200, 3, False),
            "двухсот": Numeral(200, 3, False),
            "двумстам": Numeral(200, 3, False),
            "двухстах": Numeral(200, 3, False),
            "двустам": Numeral(200, 3, False),
            "двухсотый": Numeral(200, 3, False),
            "двухсотая": Numeral(200, 3, False),
            "двухсотого": Numeral(200, 3, False),
            "двухсотой": Numeral(200, 3, False),
            "двухсотом": Numeral(200, 3, False),

            "триста": Numeral(300, 3, False),
            "трехсот": Numeral(300, 3, False),
            "тремстам": Numeral(300, 3, False),
            "трехстах": Numeral(300, 3, False),
            "трёхстах": Numeral(300, 3, False),
            "трехсотое": Numeral(300, 3, False),
            "трехсотый": Numeral(300, 3, False),
            "трехсотая": Numeral(300, 3, False),
            "трехсотого": Numeral(300, 3, False),
            "трехсотой": Numeral(300, 3, False),
            "трехсотом": Numeral(300, 3, False),
            "трёхсот": Numeral(300, 3, False),
            "трёмстам": Numeral(300, 3, False),
            "трёхсотое": Numeral(300, 3, False),
            "трёхсотый": Numeral(300, 3, False),
            "трёхсотая": Numeral(300, 3, False),
            "трёхсотого": Numeral(300, 3, False),
            "трёхсотой": Numeral(300, 3, False),
            "трёхсотом": Numeral(300, 3, False),

            "четыреста": Numeral(400, 3, False),
            "четырехсот": Numeral(400, 3, False),
            "четыремстам": Numeral(400, 3, False),
            "четырехстах": Numeral(400, 3, False),
            "четырёхстах": Numeral(400, 3, False),
            "четырехсотое": Numeral(400, 3, False),
            "четырехсотый": Numeral(400, 3, False),
            "четырехсотая": Numeral(400, 3, False),
            "четырехсотого": Numeral(400, 3, False),
            "четырехсотой": Numeral(400, 3, False),
            "четырехсотом": Numeral(400, 3, False),
            "четырёхсот": Numeral(400, 3, False),
            "четырёмстам": Numeral(400, 3, False),
            "четырёхсотое": Numeral(400, 3, False),
            "четырёхсотый": Numeral(400, 3, False),
            "четырёхсотая": Numeral(400, 3, False),
            "четырёхсотого": Numeral(400, 3, False),
            "четырёхсотой": Numeral(400, 3, False),
            "четырёхсотом": Numeral(400, 3, False),

            "пятьсот": Numeral(500, 3, False),
            "пятистах": Numeral(500, 3, False),
            "пятисот": Numeral(500, 3, False),
            "пятистам": Numeral(500, 3, False),
            "пятьсотое": Numeral(500, 3, False),
            "пятьсотый": Numeral(500, 3, False),
            "пятьсотая": Numeral(500, 3, False),
            "пятьсотого": Numeral(500, 3, False),
            "пятьсотой": Numeral(500, 3, False),
            "пятьсотом": Numeral(500, 3, False),

            "шестьсот": Numeral(600, 3, False),
            "шестистах": Numeral(600, 3, False),
            "шестисот": Numeral(600, 3, False),
            "шестистам": Numeral(600, 3, False),
            "шестисотое": Numeral(600, 3, False),
            "шестисотый": Numeral(600, 3, False),
            "шестисотая": Numeral(600, 3, False),
            "шестисотого": Numeral(600, 3, False),
            "шестисотой": Numeral(600, 3, False),
            "шестисотом": Numeral(600, 3, False),

            "семьсот": Numeral(700, 3, False),
            "семистах": Numeral(700, 3, False),
            "семисот": Numeral(700, 3, False),
            "семистам": Numeral(700, 3, False),
            "семисотое": Numeral(700, 3, False),
            "семисотый": Numeral(700, 3, False),
            "семисотая": Numeral(700, 3, False),
            "семисотого": Numeral(700, 3, False),
            "семисотой": Numeral(700, 3, False),
            "семисотом": Numeral(700, 3, False),

            "восемьсот": Numeral(800, 3, False),
            "восьмистах": Numeral(800, 3, False),
            "восьмисот": Numeral(800, 3, False),
            "восьмистам": Numeral(800, 3, False),
            "восьмисотое": Numeral(800, 3, False),
            "восьмисотый": Numeral(800, 3, False),
            "восьмисотая": Numeral(800, 3, False),
            "восьмисотого": Numeral(800, 3, False),
            "восьмисотой": Numeral(800, 3, False),
            "восьмисотом": Numeral(800, 3, False),

            "девятьсот": Numeral(900, 3, False),
            "девятистах": Numeral(900, 3, False),
            "девятисот": Numeral(900, 3, False),
            "девятистам": Numeral(900, 3, False),
            "девятьсотое": Numeral(900, 3, False),
            "девятьсотый": Numeral(900, 3, False),
            "девятьсотая": Numeral(900, 3, False),
            "девятьсотого": Numeral(900, 3, False),
            "девятьсотой": Numeral(900, 3, False),
            "девятьсотом": Numeral(900, 3, False),

            "тысяч": Numeral(1000, 4, True),
            "тысяча": Numeral(1000, 4, True),
            "тысячи": Numeral(1000, 4, True),
            "тысячном": Numeral(1000, 4, True),

            "миллион": Numeral(1000000, 5, True),
            "миллиона": Numeral(1000000, 5, True),
            "миллионов": Numeral(1000000, 5, True),
            "миллионом": Numeral(1000000, 5, True),

            "миллиард": Numeral(1000000000, 6, True),
            "миллиарда": Numeral(1000000000, 6, True),
            "миллиардов": Numeral(1000000000, 6, True),
            "миллиардном": Numeral(1000000000, 6, True),

            "триллион": Numeral(1000000000000, 7, True),
            "триллиона": Numeral(1000000000000, 7, True),
            "триллионов": Numeral(1000000000000, 7, True),
            "триллионом": Numeral(1000000000000, 7, True)
        }

        self.max_token_error = 0.3

    def get_token_sum_error_from_lists(self, token):
        token_sum_error = 0

        if isinstance(token, NumericToken):
            token_sum_error += token.error

        else:
            for sub_token in token:
                token_sum_error += self.get_token_sum_error_from_lists(sub_token)

        return token_sum_error

    def parse_tokens(self, text_line, matrix_d, level, fraction=False):
        if text_line in self.tokens.keys():
            return [NumericToken(self.tokens[text_line])]
        elif fraction:
            return [NumericToken(self.tokens_fractions[text_line])]
        else:
            return [None]

    def parse(self, text):
        text = text.strip().lower()

        if len(text) == 0:
            return ParserResult(value=0, error=1)

        # Массив для рассчета расстояния Левенштейна
        max_token_length = 13
        matrix_d = np.zeros((2, max_token_length), dtype=np.float32)

        # Разбиваем текст на токены
        raw_token_list = re.split(r"\s+", text)
        all_token_list = []
        token_list = []
        result_text_list = []
        left_space_for_number = False
        current_level = 0

        # Обрабатываем токены
        for token_idx, raw_token in enumerate(raw_token_list):
            clean_token = raw_token.strip(string.punctuation)
            current_token_list = self.parse_tokens(clean_token, matrix_d, 0)

            # Определение дробного числа:
            if clean_token in ["целых", "целой", "целым", "целая"] and token_idx != 0:

                try:
                    if raw_token_list[token_idx - 1] in self.tokens \
                            and raw_token_list[token_idx + 1] in self.tokens \
                            or raw_token_list[token_idx + 1] == "и":
                        if raw_token_list[token_idx + 2] in self.tokens_fractions \
                                or raw_token_list[token_idx + 3] in self.tokens_fractions \
                                or raw_token_list[token_idx + 4] in self.tokens_fractions:
                            current_token_list = self.parse_tokens(clean_token, matrix_d, 0, fraction=True)

                except IndexError:
                    pass

            # Обработка первого порядка:
            if clean_token in ["десятых", "десятой", "десятым", "десятая"] and token_idx != 0:

                if raw_token_list[token_idx - 1] in self.tokens:
                    current_token_list = self.parse_tokens(clean_token, matrix_d, 0, fraction=True)

            # Обработка второго порядка:
            if clean_token in ["сотых", "сотой", "сотым", "сотая"] and token_idx != 0:

                if raw_token_list[token_idx - 1] in self.tokens:
                    current_token_list = self.parse_tokens(clean_token, matrix_d, 0, fraction=True)

            # Обработка третьего порядка:
            if clean_token in ["тысячных", "тысячной", "тысячным", "тысячная"] and token_idx != 0:

                if raw_token_list[token_idx - 1] in self.tokens:
                    current_token_list = self.parse_tokens(clean_token, matrix_d, 0, fraction=True)

            # Обработка четвёртого порядка:
            if clean_token in ["десятитысячных", "десятитысячной", "десятитысячным", "десятитысячная"] and token_idx != 0:

                if raw_token_list[token_idx - 1] in self.tokens:
                    current_token_list = self.parse_tokens(clean_token, matrix_d, 0, fraction=True)

            if raw_token == "тысяча":
                if token_idx == 0 or raw_token_list[token_idx - 1] != "одна":
                    current_token_list = [NumericToken(numeral=Numeral(1, 1, False)), current_token_list[0]]
                    current_level = 1

                    if len(token_list) > 0:
                        all_token_list.append(token_list)
                        token_list = []

                    left_space_for_number = False

            if raw_token == "тысячная":
                if token_idx == 0 or raw_token_list[token_idx - 1] != "одна":
                    current_token_list = [NumericToken(numeral=Numeral(1000, 1, False))]
                    current_level = 0

                    if len(token_list) > 0:
                        all_token_list.append(token_list)
                        token_list = []

                    left_space_for_number = False

            if raw_token == "десятая":
                if token_idx == 0 or raw_token_list[token_idx - 1] != "одна":
                    current_token_list = [NumericToken(numeral=Numeral(10, 1, False))]
                    current_level = 0

                    if len(token_list) > 0:
                        all_token_list.append(token_list)
                        token_list = []

                    left_space_for_number = False

            if raw_token == "сотая":
                if token_idx == 0 or raw_token_list[token_idx - 1] != "одна":
                    current_token_list = [NumericToken(numeral=Numeral(100, 1, False))]
                    current_level = 0

                    if len(token_list) > 0:
                        all_token_list.append(token_list)
                        token_list = []

                    left_space_for_number = False

            if raw_token == "ноль":
                if token_idx != 0 or raw_token_list[token_idx - 1] == "ноль":
                    current_token_list = [NumericToken(numeral=Numeral(0, 1, False)), current_token_list[0]]
                    current_level = 0

                    if len(token_list) > 0:
                        all_token_list.append(token_list)
                        token_list = []

                    left_space_for_number = False

            if current_token_list[0] is not None:
                previous_level = current_level
                current_level = current_token_list[len(current_token_list) - 1].numeral.level
                is_eleven_to_nineteen = current_token_list[0].numeral.is_eleven_to_nineteen

                if current_level != 0 and previous_level != 0 and (
                        (previous_level < current_level <= 3) or current_level == previous_level or (current_level < previous_level <= 2 and is_eleven_to_nineteen)):
                    all_token_list.append(token_list)
                    token_list = []
                    left_space_for_number = False

            bad_tokens = True

            for current_token in current_token_list:
                if current_token is None:
                    break

                if current_token.error <= self.max_token_error:
                    bad_tokens = False
                    break

            if bad_tokens is True:
                del current_token_list
                current_token_list = [None]

            if current_token_list[0] is not None:
                if left_space_for_number is False:
                    result_text_list.append("")
                    left_space_for_number = True

                for current_token in current_token_list:
                    token_list.append(current_token)

            else:
                result_text_list.append(raw_token)
                left_space_for_number = False
                current_level = 0

                if len(token_list) > 0:
                    all_token_list.append(token_list)
                    token_list = []

        if len(token_list) > 0:
            all_token_list.append(token_list)
            token_list = []

        parser_result_list = []

        for token_list in all_token_list:
            global_level = None
            local_level = None
            global_value = None
            local_value = None
            critical_error = False

            token_count = len(token_list)

            for current_token in token_list:
                current_error = self.get_token_sum_error_from_lists(current_token)

                if current_error > self.max_token_error:
                    continue

                value = current_token.numeral.value
                level = current_token.numeral.level
                multiplier = current_token.numeral.is_multiplier

                if multiplier:
                    if global_level is None:
                        if local_level is None:
                            global_value = value
                        else:
                            global_value = np.round(local_value * value, 5)

                        global_level = level
                        local_value = None
                        local_level = None

                        current_token.is_significant = True

                    elif global_level > level:
                        if local_level is None:
                            global_value = global_value + value
                        else:
                            if value == 0.1:
                                global_value = np.round((global_value + local_value * value), 1)
                            elif value == 0.01:
                                global_value = np.round((global_value + local_value * value), 2)
                            elif value == 0.001:
                                global_value = np.round((global_value + local_value * value), 3)
                            elif value == 0.0001:
                                global_value = np.round((global_value + local_value * value), 4)

                        global_level = level
                        local_value = None
                        local_level = None

                        current_token.is_significant = True

                    else:
                        # Ошибка несоответствия уровней
                        current_token.error = 1
                        current_token.is_significant = True
                        critical_error = True

                else:
                    # Простое числительное
                    if local_level is None:
                        local_value = value
                        local_level = level

                        current_token.is_significant = True

                    elif local_level > level:
                        local_value = local_value + value
                        local_level = level

                        current_token.is_significant = True

                    else:
                        # Ошибка несоответствия уровней
                        current_token.error = 1
                        current_token.is_significant = True
                        critical_error = True

            # Считаем общий уровень ошибки
            if token_count == 0:
                total_error = 1

            else:
                total_error = 0
                significant_token_count = 0

                for current_token in token_list:
                    if current_token.is_significant:
                        total_error += current_token.error
                        significant_token_count += 1

                total_error /= significant_token_count

            if critical_error:
                # Имела место критическая ошибка
                if total_error >= 0.5:
                    total_error = 1

                else:
                    total_error *= 2

            result_value = 0

            if global_value is not None:
                result_value += global_value

            if local_value is not None:
                result_value += local_value
            parser_result_list.append(ParserResult(result_value, total_error))

        return parser_result_list, result_text_list
