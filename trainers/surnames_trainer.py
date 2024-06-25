from configs.app_config import CREATORS_FOLDER_PATH
from configs.training_config import *
from utils.files_helper import *
from utils.embeddings import embed_name, adjust_creation
from vae.dataset import NamesDataset
from vae.assembly import VAE
from vae.loss_function import loss_function
from vae.temperature import create_name
import os
import torch
from torch.utils.data import DataLoader


class SurnamesTrainer:

    def __init__(self):
        self.names = read_unique_names('surnames')
        self.vae_path = os.path.join(CREATORS_FOLDER_PATH, 'surnames.pth')

    def train(self):
        # Use embedded names for dataset.
        embedded_names = [embed_name(name) for name in self.names]
        max_len = max(len(name) for name in embedded_names)
        update_max_len({"surnames": max_len})

        dataset = NamesDataset(embedded_names, max_len)
        save_dataset('surnames', dataset)

        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        vae = VAE(
            input_dim=max_len,
            max_len=max_len,
            features=len(dataset.encoder.classes_),
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

                reconstruction_batch, mu, log_var = vae(batch.float())
                loss = loss_function(reconstruction_batch, batch, mu, log_var)
                loss.backward()

                if epoch % DISPLAY_FREQ == 0:
                    epoch_loss += loss.item()

                optimizer.step()

            if epoch % DISPLAY_FREQ == 0:
                print(f'Epoch {epoch} Loss: {epoch_loss / len(dataset)}')
                epoch_loss -= epoch_loss

        torch.save(vae.state_dict(), self.vae_path)

    def evaluate(self, num_names: int, temperature: float):
        max_len = read_max_len('surnames')
        dataset = read_dataset('surnames')

        vae = VAE(
            input_dim=max_len,
            max_len=max_len,
            features=len(dataset.encoder.classes_),
        )

        try:  # If pre-trained VAE is found.
            vae.load_state_dict(torch.load(self.vae_path))

        except FileNotFoundError:
            raise FileNotFoundError('No pre-trained VAE found.')

        vae.eval()
        creations = []
        while len(creations) < num_names:
            creation = create_name(
                vae, max_len, dataset.encoder, temperature
            )
            creation = adjust_creation(creation)
            if creation not in creations + self.names:  # Ensure distinct creation.
                creations.append(creation)

        return ', '.join(creations)


if __name__ == '__main__':
    trainer = SurnamesTrainer()
    # trainer.train()
    print(trainer.evaluate(20, 0.05))
