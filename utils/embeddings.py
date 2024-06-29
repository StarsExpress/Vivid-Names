from configs.names_config import EMBED_CHARS_DICT


start_char, end_char = EMBED_CHARS_DICT['start'], EMBED_CHARS_DICT['end']


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
    Step 1: remove any embedded start_char & end_char.

    Step 2: capitalize 1st letter. If 1st two chars are Mc, also capitalize 3rd char.

    Args:
        creation (str): created name.

    Returns:
        str: adjusted name.
    """
    creation = creation.replace(start_char, '').replace(end_char, '').capitalize()

    if creation[:2] == 'Mc':
        creation = creation[:2] + creation[2:].capitalize()
    return creation
