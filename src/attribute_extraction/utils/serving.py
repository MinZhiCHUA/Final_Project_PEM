from typing import Any, List, Optional, Tuple

import pandas as pd

# from attribute_extraction.config import LANG_QUESTION_MAPPING
from attribute_extraction.data.processing import clean_html_text


# def generate_question(lang: str, attribute_label: str) -> str:
#     return f"{LANG_QUESTION_MAPPING[lang.upper()]} {attribute_label}?"


def generate_context(title: str, desc: str) -> str:
    return clean_html_text(f"{title} {desc}")


def get_lov_values(lov_dict: dict, attribute_code: str, lang: str) -> Optional[list]:
    """This function gets the LOV values associated to a given attribute_code and lang.

    Args:
        lov_dict ([type]): the dict generated thanks to load_lov_dict()
        attribute_code (str): STEP attribute code. eg. "21148"
        lang (str): the lang code. eg. "FR"

    Returns:
        Optional[list]: returns the list of values if exists, else return None
    """
    lov_values = lov_dict.get(f"('{attribute_code}', '{lang}')", {}).get("lov_values")
    return lov_values


def generate_context_metadata(
    title: str, desc: str, metadata: Optional[List[Tuple[str, Optional[Any]]]] = None
) -> str:

    context = f"{title} {desc}"
    if metadata:
        metadata_str = [
            f"{x[0]} {x[1]}"
            for x in metadata
            if not (x[1] is None or pd.isna(x[1]) or not str(x[1]).strip())
        ]

        context = context + " " + " ".join(metadata_str)

    return clean_html_text(context)