"""All names configurations."""

import string

# Embeddings settings.
LOWER_CASE_LIST = list(string.ascii_lowercase)
UPPER_CASE_LIST = list(string.ascii_uppercase)

EMBED_CHARS_DICT = {
    'start': '~',  # At left edge to represent name's start.
    'end': '#',  # At right edge to represent name's end.
}
