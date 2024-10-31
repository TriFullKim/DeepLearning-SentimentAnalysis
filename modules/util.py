import pickle
import gzip
import datetime
import json
import random


def jsonify(s):
    try:
        return json.loads(s)
    except:
        return s


def json_to_str(obj):
    return json.dumps(obj, ensure_ascii=False)


def have_key(dict: dict, key: str):
    return key in dict.keys()


def time_gap(mu=6, sigma=2):
    delta = int(random.normalvariate(mu, sigma))
    return delta if delta > 0 else 1


def now():
    return datetime.datetime.now()


def save_json(path, json_obj):
    with open(path, "w") as f:
        json.dump(json_obj, f, ensure_ascii=False)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_pkl(path, obj):
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def is_symbol(char: str) -> bool:
    """
    ASCII코드 중에서 숫자/영문이 아닌 것인지 확인합니다. = 특수문자인지 판단합니다.
    """
    return char.isascii() and not char.isalnum()


def char_to_hex_ascii(char) -> str:
    """
    "문자"를 "16진수ASCII코드"로 변환합니다.
    URI에서는 특정한 특수문자들이 사용이 불가능 하므로, HEX로 변환된 ASCII코드를 사용합니다.

    Returns:
        str: "%"+"16진수ASCII코드"
    """
    return hex(ord(char)).upper().replace("0X", "%")


def hex_ascii_to_char(hex_ascii) -> str:
    """
    "16진수ASCII코드"를 "문자"로 변환합니다.
    """
    return chr(int(hex_ascii.replace("%", "0X").lower(), 16))


def ascii_transformer(str: str):
    """
    특수문자가 있는 "문자열"를 "16진수ASCII코드"로 변환합니다.
    """
    result_str = []
    for char in [*str]:
        if is_symbol(char):
            result_str.append(char_to_hex_ascii(char))
        elif not is_symbol(char):
            result_str.append(char)

    return "".join(result_str)


def char_transformer(str: str):
    """
    "16진수ASCII코드"가 있는 "문자열"를 특수문자로 변환합니다.
    """
    result_str = []

    SKIP_FACTOR = 0
    for i, char in enumerate([*str]):
        if SKIP_FACTOR > 0:
            SKIP_FACTOR -= 1
            continue
        if char == "%" and i + 2 < len(str):
            SKIP_FACTOR = 2
            hex_ascii = str[i] + str[i + 1] + str[i + 2]
            result_str.append(hex_ascii_to_char(hex_ascii))
        else:
            result_str.append(char)
    return "".join(result_str)


def decompose_url_query(url):
    squeezed_query = url.split("?")[-1]
    unsqueezed_query = {
        parser.split("=")[0]: parser.split("=")[1] if len(parser.split("=")) > 1 else ""
        for parser in squeezed_query.split("&")
    }
    return unsqueezed_query


def composing_url_query(query):
    return "&".join([str(key) + "=" + str(val) for key, val in query.items()])
