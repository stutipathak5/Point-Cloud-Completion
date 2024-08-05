from sklearn.model_selection import train_test_split
from utils import CatenaryDataset
from torch.utils.data import Subset, DataLoader
from models.vqvae import VQVAE
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CatenaryDataset(gt_path = 'data\difficult\complete.npy', partial_path = 'data/difficult/partial.npy', voxel_size = 0.05, grid_size =(20, 20, 20), transform = None)


print(len(dataset))
train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# print(list(train_loader)[0].size())
# print(list(train_loader)[0][0].size(), list(train_loader)[0][1].size(), list(train_loader)[0][2].size(), list(train_loader)[0][3].size())

model = VQVAE(128, 32, 2, 512, 64, 0.25).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4, amsgrad=True)

model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}

r_loss = nn.BCELoss()
log_interval = 10
def train():
    num_epochs = 100
    for i in range(num_epochs):
        print(i)
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()

            embedding_loss, x_hat, perplexity = model(x)
            print(x.size(), x_hat.size())
            # new
            # recon_loss = r_loss(x, x_hat)
            loss = recon_loss + embedding_loss

            loss.backward()
            optimizer.step()

            results["recon_errors"].append(recon_loss.cpu().detach().numpy())
            results["perplexities"].append(perplexity.cpu().detach().numpy())
            results["loss_vals"].append(loss.cpu().detach().numpy())
            results["n_updates"] = i

            if i % log_interval == 0:
                """
                save model and print values
                """
                # if args.save:
                #     hyperparameters = args.__dict__
                #     utils.save_model_and_results(
                #         model, results, hyperparameters, args.filename)

                print('Update #', i, 'Recon Error:',
                    np.mean(results["recon_errors"][-log_interval:]),
                    'Loss', np.mean(results["loss_vals"][-log_interval:]),
                    'Perplexity:', np.mean(results["perplexities"][-log_interval:]))



















# def train():

#     for i in range(5000):
#         (x, _) = next(iter(train_loader))
#         x = x.to(device)
#         optimizer.zero_grad()

#         embedding_loss, x_hat, perplexity = model(x)
#         recon_loss = torch.mean((x_hat - x)**2) / x_train_var
#         loss = recon_loss + embedding_loss

#         loss.backward()
#         optimizer.step()

#         results["recon_errors"].append(recon_loss.cpu().detach().numpy())
#         results["perplexities"].append(perplexity.cpu().detach().numpy())
#         results["loss_vals"].append(loss.cpu().detach().numpy())
#         results["n_updates"] = i

#         if i % args.log_interval == 0:
#             """
#             save model and print values
#             """
#             if args.save:
#                 hyperparameters = args.__dict__
#                 utils.save_model_and_results(
#                     model, results, hyperparameters, args.filename)

#             print('Update #', i, 'Recon Error:',
#                   np.mean(results["recon_errors"][-args.log_interval:]),
#                   'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
#                   'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]))


if __name__ == "__main__":
    train()
