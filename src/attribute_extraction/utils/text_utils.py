import re

CLEANER = re.compile("<.{1,27}?>")


def clean_html(raw_html):
    cleantext = re.sub(CLEANER, "", raw_html)
    return cleantext.strip()
