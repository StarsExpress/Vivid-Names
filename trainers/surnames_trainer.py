from configs.app_config import CREATORS_FOLDER_PATH
from configs.training_config import *
from utils.files_helper import read_unique_names
from dataset import NamesDataset
from vae.assembly import Assembly
from loss_function import loss_function
from temperature import create_name
import os
import torch
from torch.utils.data import DataLoader


class SurnamesTrainer:

    def __init__(self):
        self.names = read_unique_names('surnames').tolist()
        self.max_len = max(len(name) for name in self.names)
        self.dataset = NamesDataset(self.names, self.max_len)
        self.vae_path = os.path.join(CREATORS_FOLDER_PATH, 'surnames.pth')

    def train(self):
        dataloader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)

        vae = Assembly(
            input_dim=self.max_len,
            hidden_dim=256,
            latent_dim=20,
            max_len=self.max_len,
            vocab_size=len(self.dataset.encoder.classes_),
        )

        try:
            vae.load_state_dict(torch.load(self.vae_path))

        except FileNotFoundError:
            pass

        optimizer = torch.optim.Adam(
            vae.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=BETAS,
        )

        epoch_loss = 0
        for epoch in range(1, EPOCHS + 1):
            vae.train()

            for batch in dataloader:
                optimizer.zero_grad()
                recon_batch, mu, log_var = vae(batch.float())
                loss = loss_function(recon_batch, batch, mu, log_var)
                loss.backward()

                if epoch % DISPLAY_FREQ == 0:
                    epoch_loss += loss.item()

                optimizer.step()

            if epoch % DISPLAY_FREQ == 0:
                print(f'Epoch {epoch} Loss: {epoch_loss / len(self.dataset)}.')
                epoch_loss -= epoch_loss

        torch.save(vae.state_dict(), self.vae_path)

    def evaluate(self, num_names: int, temperature: float):
        latent_dim = 20

        vae = Assembly(
            input_dim=self.max_len,
            hidden_dim=256,
            latent_dim=20,
            max_len=self.max_len,
            vocab_size=len(self.dataset.encoder.classes_),
        )

        try:  # If pre-trained VAE is found.
            vae.load_state_dict(torch.load(self.vae_path))

        except FileNotFoundError:
            raise FileNotFoundError('No pre-trained VAE found.')

        vae.eval()  # Go to evaluation mode.

        creations = []
        while len(creations) < num_names:
            creation = create_name(
                vae, latent_dim, self.max_len, self.dataset.encoder, temperature
            )
            if creation not in self.names + creations:  # Only accept non-existing names.
                creations.append(creation)

        return ', '.join(creations)


if __name__ == '__main__':
    trainer = SurnamesTrainer()
    # trainer.train()
    print(trainer.evaluate(5, 0.3))
