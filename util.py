import string
import easyocr
import math
# Initialize the OCR reader
reader = easyocr.Reader(['en'])

# Mapping dictionaries for character conversion
dict_char_to_int = {
    'O': '0',
    'I': '1',
    'J': '3',
    'A': '4',
    'G': '6',
    'S': '5',
    'Q': '0',
}

dict_int_to_char = {
    '0': 'O',
    '1': 'I',
    '3': 'J',
    '4': 'A',
    '6': 'G',
    '8': 'B'
}

jeepney_codes = [
    "01K", "01B", "01C", "02B", "03A", "03B", "03L", "03Q", "04B", "04C", "04D", "04I", "04L",
    "04M", "04H", "06B", "06C", "06H", "06F", "06G", "07B", "08G", "09C", "09F", "09G", "09H",
    "10G", "10M", "10H", "10F", "11A", "12G", "12I", "12L", "12D", "13B", "13C", "13H", "14D",
    "15", "17D", "17B", "17C", "20A", "20B", "21A", "21D", "22A", "22D", "22G", "22I", "23D",
    "23", "24", "25", "62B", "62C"
]


def jeepcode_complies_format(text):
    # Check if the length of the text is either 2 or 3
    if len(text) not in [2, 3]:
        return False

    # Check if the first two characters are digits
    if not (text[0].isdigit() and text[1].isdigit()):
        return False

    # If the length is 3, check if the third character is an uppercase letter
    if len(text) == 3 and text[2] not in string.ascii_uppercase:
        return False

    return True


def format_jeepcode(text):
    # Check if the length of the text is either 2 or 3
    if len(text) not in [2, 3]:
        return text  # Return the original text if it does not comply with the length requirement

    jeep_code = ''
    # Define mappings for each character position
    mapping = {
        0: dict_char_to_int,  # First character conversion
        1: dict_char_to_int,  # Second character conversion
        2: dict_int_to_char  # Third character conversion (only if length is 3)
    }

    # Convert each character according to the mapping
    for i in range(len(text)):
        if text[i] in mapping[i].keys():
            jeep_code += mapping[i][text[i]]
        else:
            jeep_code += text[i]  # Append the character as is if no conversion is found

    return jeep_code


def read_jeepcode(jeepcode_crop):
    detections = reader.readtext(jeepcode_crop)
    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')
        formatted_text = format_jeepcode(text)
        # Check if the formatted text complies with the jeep code format and is in the list of valid codes
        if jeepcode_complies_format(formatted_text) and formatted_text in jeepney_codes:
            print(f"Jeepney Code: {formatted_text}", f"Percent: {math.ceil(score * 100) / 100}%")
            return formatted_text, score

    return None, None

