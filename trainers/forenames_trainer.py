from configs.paths_config import MODELS_FOLDER_PATH
from configs.training_config import *
from utils.files_helper import *
from utils.embeddings import embed_name, adjust_creation
from vae.dataset import NamesDataset
from vae.assembly import VAE
from vae.loss import compute_loss
from vae.temperature import create_name
import os
import torch
from torch.utils.data import DataLoader


class ForenamesTrainer:
    """
    Train and evaluate VAE model on forenames given gender.

    Attributes:
        names (list): list of unique forenames.
        vae_path (str): path to VAE model.

    Methods:
        train(): train model on forenames.
        evaluate(num_names: int, temperature: float): create a number of new names.
    """

    def __init__(self, gender):
        """
        Initialize with list of unique forenames and path to VAE model.
        """
        self.gender = gender
        self.names = read_unique_names(f"{gender}_forenames")
        self.vae_path = os.path.join(MODELS_FOLDER_PATH, f"{gender}_forenames.pth")

    def train(self, epochs: int):
        """
        Train VAE on forenames given number of epochs. Use embedded names for dataset.
        """
        embedded_names = [embed_name(name) for name in self.names]
        # Timesteps must be length of the longest name.
        timesteps = max(len(name) for name in embedded_names)
        update_timesteps({f"{self.gender}_forenames": timesteps})

        dataset = NamesDataset(embedded_names, timesteps)
        save_dataset(f"{self.gender}_forenames", dataset)

        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        vae = VAE(
            timesteps=timesteps,
            features=len(dataset.encoder.classes_),
            name_type=f"{self.gender}_forenames",
        )

        try:
            vae.load_state_dict(torch.load(self.vae_path))

        except FileNotFoundError:
            pass

        optimizer = torch.optim.AdamW(
            vae.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=BETAS,
        )

        epoch_loss = 0
        for epoch in range(1, epochs + 1):
            vae.train()
            for batch in dataloader:
                optimizer.zero_grad()

                reconstructed_batch, latent_mean, latent_log_var = vae(batch.float())
                loss = compute_loss(
                    reconstructed_batch,
                    batch,
                    latent_mean,
                    latent_log_var,
                    f"{self.gender}_forenames",
                )
                loss.backward()

                if epoch % DISPLAY_FREQ == 0:
                    epoch_loss += loss.item()

                optimizer.step()

            if epoch % DISPLAY_FREQ == 0:
                print(f"Epoch {epoch} Loss: {epoch_loss / len(dataset)}")
                epoch_loss -= epoch_loss

        torch.save(vae.state_dict(), self.vae_path)

    def evaluate(self, num_names: int, temperature: float):
        """
        Create a number of new forenames for evaluation.

        Args:
            num_names (int): number of new names to create.
            temperature (float): creativity level. Higher means more creative/randomness.

        Returns:
            str: string of all created names, separated by commas.
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
            creation = create_name(
                vae,
                dataset.encoder,
                temperature,
                f"{self.gender}_forenames",
            )
            creation = adjust_creation(creation)
            if creation not in creations + self.names:  # Ensure distinct creation.
                creations.append(creation)

        print(f"\n{temperature} Temperature:")
        return ", ".join(creations)


if __name__ == "__main__":
    trainer = ForenamesTrainer("female")
    # trainer.train(300)

    num_creations, low_temperature, high_temperature = 20, 0.01, 0.1
    print(trainer.evaluate(num_creations, low_temperature))
    print(trainer.evaluate(num_creations, high_temperature))
