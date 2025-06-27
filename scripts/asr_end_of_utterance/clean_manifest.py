# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example usage:

```bash
python clean_manifest.py \
    /path/to/manifest/dir \
    -o /path/to/output/dir
```

"""

import argparse
import re
import unicodedata
from pathlib import Path
from string import punctuation

from num2words import num2words
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest

punctuations = punctuation.replace("'", "")

text_normalizer = EnglishTextNormalizer()

parser = argparse.ArgumentParser(description="Clean manifest file by droping PnC")
parser.add_argument(
    "input_manifest",
    type=str,
    help="Path to the input manifest file to be cleaned.",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=None,
    help="Path to the output manifest file after cleaning.",
)
parser.add_argument(
    "-lower",
    "--lowercase",
    type=bool,
    default=False,
    help="Whether to convert the text to lowercase.",
)
parser.add_argument(
    "-drop",
    "--remove_punc",
    type=bool,
    default=False,
    help="Whether to remove punctuation from the text.",
)
parser.add_argument(
    "--normalize",
    type=bool,
    default=False,
    help="Whether to normalize the text using Whisper text normalizer.",
)
parser.add_argument(
    "-n2w",
    "--replace_numbers",
    type=bool,
    default=True,
    help="Whether to replace numbers with words.",
)
parser.add_argument(
    "-p",
    "--pattern",
    type=str,
    default="**/*.json",
    help="Pattern to match files in the input directory.",
)
parser.add_argument(
    "-t",
    "--text_field",
    type=str,
    default="text",
    help="Field in the manifest to clean. Default is 'text'.",
)
parser.add_argument(
    "--auto_pc",
    action="store_true",
    help="If set, will add auto capitalization and punctuation at the end of the text.",
)


def convert_to_spoken(text: str) -> str:
    # Mapping of metric units to spoken forms
    unit_map = {
        "kg": "kilograms",
        "g": "grams",
        "mg": "milligrams",
        "l": "liters",
        "ml": "milliliters",
        "cm": "centimeters",
        "mm": "millimeters",
        "m": "meters",
        "km": "kilometers",
        "°c": "degrees celsius",
        "°f": "degrees fahrenheit",
        "oz": "ounces",
        "lb": "pounds",
        "lbs": "pounds",
    }

    # Replace metric units like "12kg" or "5 ml"
    def replace_metric(match):
        number = match.group(1)
        unit = match.group(2).lower()
        spoken_unit = unit_map.get(unit, unit)
        return f"{number} {spoken_unit}"

    # Replace time like "5am" or "6PM"
    def replace_ampm(match):
        hour = match.group(1)
        meridiem = match.group(2).lower()
        return f"{hour} {'a m' if meridiem == 'am' else 'p m'}"

    # Replace time like "1:30pm"
    def replace_colon_time(match):
        hour = match.group(1)
        minute = match.group(2)
        meridiem = match.group(3).lower()
        return f"{hour} {minute} {'a m' if meridiem == 'am' else 'p m'}"

    # Convert feet and inches like 5'11" to "5 feet 11 inches"
    def replace_feet_inches(match):
        feet = match.group(1)
        inches = match.group(2)
        return f"{feet} feet {inches} inches"

    # Convert just feet (e.g., 6') to "6 feet"
    def replace_feet_only(match):
        feet = match.group(1)
        return f"{feet} feet"

    # Convert just inches (e.g., 10") to "10 inches"
    def replace_inches_only(match):
        inches = match.group(1)
        return f"{inches} inches"

    # Apply replacements
    # First: time with colon (e.g., 1:30pm)
    text = re.sub(r'\b(\d{1,2}):(\d{2})(am|pm)\b', replace_colon_time, text, flags=re.IGNORECASE)

    # Then: basic am/pm (e.g., 5am)
    text = re.sub(r'\b(\d{1,2})(am|pm)\b', replace_ampm, text, flags=re.IGNORECASE)

    # Then: replace 1st, 2nd, 3rd, etc
    text = text.replace("1st", "first")
    text = text.replace("2nd", "second")
    text = text.replace("3rd", "third")
    text = text.replace("@", " at ")

    # Finally: metric units
    text = re.sub(
        r'\b(\d+(?:\.\d+)?)\s?(kg|g|mg|l|ml|cm|mm|m|km|°c|°f|oz|lbs?|LB|LBS?)\b',
        replace_metric,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r'\b(\d+)\'(\d+)"', replace_feet_inches, text)  # e.g., 5'11"
    text = re.sub(r'\b(\d+)\'', replace_feet_only, text)  # e.g., 6'
    text = re.sub(r'(\d+)"', replace_inches_only, text)  # e.g., 10"

    return text


def replace_numbers_with_words(text):
    def convert_number(match):
        num_str = match.group()
        original = num_str

        # Remove dollar sign
        is_dollar = False
        if num_str.startswith('$'):
            is_dollar = True
            num_str = num_str[1:]

        # Remove commas
        num_str = num_str.replace(',', '')

        try:
            if '.' in num_str:
                # Convert decimal number
                integer_part, decimal_part = num_str.split('.')
                words = num2words(int(integer_part)) + ' point ' + ' '.join(num2words(int(d)) for d in decimal_part)
            else:
                words = num2words(int(num_str))
            if is_dollar:
                words += ' dollars'
            return words + " "
        except:
            return original  # Return original if conversion fails

    # Pattern matches: $3,000 or 3,000.45 or 1234
    pattern = re.compile(r'\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\$?\d+(?:\.\d+)?')
    result = pattern.sub(convert_number, text)

    result = " ".join(result.split())  # Remove extra spaces
    return result


def drop_punctuations(text):
    """
    Clean the text by removing invalid characters and converting to lowercase.

    :param text: Input text.
    :return: Cleaned text.
    """
    valid_chars = "abcdefghijklmnopqrstuvwxyz'"
    text = ''.join([c for c in text if c in valid_chars or c.isspace() or c == "'"])
    text = ' '.join(text.split())  # Remove extra spaces
    return text.strip()


def clean_label(_str: str) -> str:
    """
    Remove unauthorized characters in a string, lower it and remove unneeded spaces
    """
    # replace_with_space = [char for char in '/?*\",.:=?_{|}~¨«·»¡¿„…‧‹›≪≫!:;ː→']
    replace_with_blank = [char for char in '`¨´‘’“”`ʻ‘’“"‘”']
    replace_with_apos = [char for char in '‘’ʻ‘’‘']
    _str = _str.strip()
    for i in replace_with_blank:
        _str = _str.replace(i, "")
    for i in replace_with_apos:
        _str = _str.replace(i, "'")

    text = _str
    text = text.replace("\u2103", "celsius")
    text = text.replace("\u2109", "fahrenheit")
    text = text.replace("\u00b0", "degrees")
    text = text.replace("\u2019", "'")
    text = text.replace("\\", ".")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    ret = " ".join(_str.split())
    return ret


def ends_with_punctuation(s: str) -> bool:
    # Strip trailing whitespace
    s = s.rstrip()

    # consider this set to be punctuation that's acceptable to end a sentence with
    puncturation_chars = [",", ".", ":", ";", "?", "!", "-", "—", "–", "…"]

    # If string is empty after stripping, return False
    if not s:
        return False

    # Get the last character
    last_char = s[-1]

    # Return True if the last character is punctuation, otherwise False
    return last_char in puncturation_chars


def add_period_if_needed(text: str) -> str:
    """
    Add a period at the end of the text if it does not already end with one.
    """
    if not ends_with_punctuation(text):
        text += "."
    return text.strip()


def capitalize_self_i(text):
    # Replace standalone lowercase "i" with "I"
    # Handles "i", "i.", "i?", "i'll", "i'm", etc.
    return re.sub(r'\b(i)(?=[\s.,!?;:\'\"-]|$)', r'I', text)


def add_space_after_punctuation(text):
    # Add a space after punctuation if it's not already followed by one or by the end of the string
    return re.sub(r'([,\.?;:])(?=\S)', r'\1 ', text)


def add_auto_capitalization(text):
    if text.lower() != text:
        # If the text is not all lowercase, we assume it has some capitalization
        return text

    # Remove space before punctuation (.,!?;:)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)

    # Capitalize the first letter of each sentence
    def capitalize_sentences(match):
        return match.group(1) + match.group(2).upper()

    # Ensure first character is capitalized
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]

    text = capitalize_self_i(text)
    text = add_space_after_punctuation(text)
    # Capitalize after sentence-ending punctuation followed by space(s)
    text = re.sub(r'([.!?]\s+)([a-z])', capitalize_sentences, text)
    return text


def unicode_to_ascii(text: str) -> str:
    """
    Converts text with accented or special Latin characters (e.g., ó, ñ, ū, ō)
    into their closest ASCII equivalents.
    """
    # Normalize the string to NFKD to separate base characters from diacritics
    normalized = unicodedata.normalize('NFKD', text)

    # Encode to ASCII bytes, ignoring characters that can't be converted
    ascii_bytes = normalized.encode('ascii', 'ignore')

    # Decode back to string
    ascii_text = ascii_bytes.decode('ascii')

    return ascii_text


def main(args):
    text_field = args.text_field
    manifest_files = Path(args.input_manifest)
    if manifest_files.is_dir():
        manifest_files = list(manifest_files.glob(args.pattern))
    elif manifest_files.is_file():
        manifest_files = [manifest_files]
    else:
        raise ValueError(f"Invalid input manifest path: {args.input_manifest}")

    for manifest_file in manifest_files:
        print(f"Processing manifest file: {manifest_file}")
        postfix = "-cleaned"
        postfix += "_norm" if args.normalize else ""
        postfix += "_n2w" if args.replace_numbers else ""
        if args.lowercase and args.remove_punc:
            postfix += "_noPC"
        else:
            postfix += "_lc" if args.lowercase else ""
            postfix += "_np" if args.remove_punc else ""
        postfix += "_aPC" if args.auto_pc else ""

        output_manifest = manifest_file.with_name(f"{manifest_file.stem}{postfix}{manifest_file.suffix}")

        if args.output:
            if args.output.endswith(".json"):
                if len(manifest_files) > 1:
                    raise ValueError("Output path must be a directory when processing multiple manifest files.")
                output_manifest = Path(args.output)
            else:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_manifest = output_dir / output_manifest.name

        manifest = read_manifest(str(manifest_file))

        for i, item in enumerate(manifest):
            text = str(item[text_field])
            manifest[i]["original_text"] = text
            text = unicode_to_ascii(text)
            if args.normalize:
                text = text_normalizer(text)
            if args.replace_numbers:
                text = convert_to_spoken(text)
                text = replace_numbers_with_words(text)
            if args.lowercase:
                text = text.lower()
            if args.remove_punc:
                text = text.translate(str.maketrans("", "", punctuations))
                text = drop_punctuations(text)
            if args.auto_pc:
                text = add_auto_capitalization(text)
            manifest[i]["text"] = clean_label(text)

        write_manifest(str(output_manifest), manifest)
        print(f"Cleaned manifest saved to {output_manifest}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
