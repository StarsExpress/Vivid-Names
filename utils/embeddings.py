from configs.names_config import EMBED_CHARS_DICT


start_char = EMBED_CHARS_DICT['start']
end_char = EMBED_CHARS_DICT['end']


def embed_name(name: str):
    """
    Add start_char at front and end_char at back.

    Args:
        name (str): name to embed.

    Returns:
        str: embedded name.
    """
    return f"{start_char}{name}{end_char}"


def adjust_creation(creation: str):
    """
    Remove any embedded start_char & end_char. Capitalize 1st letter.

    Args:
        creation (str): created name.

    Returns:
        str: adjusted name.
    """
    creation = creation.replace(start_char, '').replace(end_char, '')
    return creation.capitalize()
