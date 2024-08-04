import os
import torch
from configs.paths_config import MODELS_FOLDER_PATH
from utils.files_helper import read_unique_names, read_timesteps, read_dataset
from utils.embeddings import adjust_creation
from vae.assembly import VAE
from vae.temperature import create_name


class ForenamesCreator:
    """
    Create new forenames via pre-trained VAE model.

    Attributes:
        gender (str): gender of forenames to be created.
        names (list): list of unique forenames of given gender.
        vae_path (str): path to VAE model.

    Methods:
        create(num_names: int, temperature: float): create a number of new forenames.
    """

    def __init__(self, gender: str):
        """
        Initialize with gender.

        Args:
            gender (str): gender of forenames to be created.
        """
        self.gender = gender
        self.names = read_unique_names(f"{gender}_forenames")
        self.vae_path = os.path.join(MODELS_FOLDER_PATH, f"{gender}_forenames.pth")

    def create(self, num_names: int, temperature: float):
        """
        Create a number of new forenames for production.

        Args:
            num_names (int): number of new names to create.
            temperature (float): creativity level. Higher means more creative/randomness.

        Returns:
            list: list of all created names.
        """
        timesteps = read_timesteps(f"{self.gender}_forenames")
        dataset = read_dataset(f"{self.gender}_forenames")

        vae = VAE(
            timesteps=timesteps,
            features=len(dataset.encoder.classes_),
            name_type=f"{self.gender}_forenames",
        )

        try:  # If pre-trained VAE is found.
            vae.load_state_dict(torch.load(self.vae_path))

        except FileNotFoundError:
            raise FileNotFoundError("No pre-trained VAE found.")

        vae.eval()
        creations = []
        while len(creations) < num_names:
            creation = create_name(vae, dataset.encoder, temperature, f"{self.gender}_forenames")
            creation = adjust_creation(creation)
            if creation not in creations + self.names:  # Ensure distinct creation.
                creations.append(creation)

        return creations


if __name__ == "__main__":
    creator = ForenamesCreator("female")
    print(creator.create(20, 0.2))
