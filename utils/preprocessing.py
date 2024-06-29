from configs.paths_config import DATA_FOLDER_PATH
from configs.pages_config import CREATIVITY_DICT
import os
import base64


def process_image_body():
    """
    Read and encode image to base64. Prepare it to be used as a background image in a style tag.

    Returns:
        str: string containing style tag with background image.
    """
    image_path = os.path.join(DATA_FOLDER_PATH, 'background.jpg')
    image = open(image_path, 'rb')
    background_image = image.read()
    image.close()

    decoded_image = base64.b64encode(background_image).decode()
    page_image = f"""
                    <style>
                    .stApp {{
                    background-image: url("data:image/jpg;base64,{decoded_image}");
                    background-size: cover;
                    }}
                    </style>
                  """
    return page_image


def process_temperature(display_temperature: float):
    """
    Convert display temperature to real temperature used in VAE model.

    Args:
        display_temperature (float): selected temperature as displayed in UI.

    Returns:
        float: real temperature to be used in VAE model.
    """
    ratio = display_temperature - CREATIVITY_DICT["display_min"]
    ratio /= (CREATIVITY_DICT["display_max"] - CREATIVITY_DICT["display_min"])
    return ratio * CREATIVITY_DICT["real_max"] + (1 - ratio) * CREATIVITY_DICT["real_min"]
