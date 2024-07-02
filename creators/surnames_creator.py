from configs.paths_config import MODELS_FOLDER_PATH
from utils.files_helper import *
from utils.embeddings import adjust_creation
from vae.assembly import VAE
from vae.temperature import create_name
import os
import torch


class SurnamesCreator:
    """
    Create new surnames via pre-trained VAE model.

    Attributes:
        names (list): list of unique surnames.
        vae_path (str): path to VAE model.

    Methods:
        create(num_names: int, temperature: float): create a number of new surnames.
    """

    def __init__(self):
        """
        Initialize with list of unique surnames and path to model.
        """
        self.names = read_unique_names("surnames")
        self.vae_path = os.path.join(MODELS_FOLDER_PATH, f"surnames.pth")

    def create(self, num_names: int, temperature: float):
        """
        Create a number of new surnames for production.

        Args:
            num_names (int): number of new names to create.
            temperature (float): creativity level. Higher means more creative/randomness.

        Returns:
            list: list of all created names.
        """
        max_len = read_max_len("surnames")
        dataset = read_dataset("surnames")

        vae = VAE(
            input_dim=max_len,
            max_len=max_len,
            features=len(dataset.encoder.classes_),
            names_type="surnames",
        )

        try:  # If pre-trained VAE is found.
            vae.load_state_dict(torch.load(self.vae_path))

        except FileNotFoundError:
            raise FileNotFoundError("No pre-trained VAE found.")

        vae.eval()
        creations = []
        while len(creations) < num_names:
            creation = create_name(vae, dataset.encoder, temperature, "surnames")
            creation = adjust_creation(creation)
            if creation not in creations + self.names:  # Ensure distinct creation.
                creations.append(creation)

        return creations


if __name__ == "__main__":
    creator = SurnamesCreator()
    print(creator.create(20, 0.1))
