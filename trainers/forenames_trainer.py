from configs.app_config import CREATORS_FOLDER_PATH
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

    def __init__(self, gender: str):
        self.gender = gender
        self.names = read_unique_names(f"{gender}_forenames")
        self.vae_path = os.path.join(CREATORS_FOLDER_PATH, f"{gender}_forenames.pth")

    def train(self):
        # Use embedded names for dataset.
        embedded_names = [embed_name(name) for name in self.names]
        max_len = max(len(name) for name in embedded_names)
        update_max_len({f"{self.gender}_forenames": max_len})

        dataset = NamesDataset(embedded_names, max_len)
        save_dataset(f"{self.gender}_forenames", dataset)

        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        vae = VAE(
            input_dim=max_len,
            max_len=max_len,
            features=len(dataset.encoder.classes_),
            names_type=f"{self.gender}_forenames",
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

        vae.train()
        epoch_loss = 0
        for epoch in range(1, EPOCHS + 1):
            for batch in dataloader:
                optimizer.zero_grad()

                reconstructed_batch, latent_mean, latent_log_var = vae(batch.float())
                loss = compute_loss(
                    reconstructed_batch, batch, latent_mean, latent_log_var, f"{self.gender}_forenames",
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
        max_len = read_max_len(f"{self.gender}_forenames")
        dataset = read_dataset(f"{self.gender}_forenames")

        vae = VAE(
            input_dim=max_len,
            max_len=max_len,
            features=len(dataset.encoder.classes_),
            names_type=f"{self.gender}_forenames",
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

        return ", ".join(creations)


if __name__ == "__main__":
    trainer = ForenamesTrainer("male")
    # trainer.train()
    print(trainer.evaluate(20, 0.2))
