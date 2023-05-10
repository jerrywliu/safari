# make sure correct hyena sequence model is in registry before import
from train import SequenceLightningModule
import torch
import matplotlib.pyplot as plt

from src.dataloaders.icl_pde import PDEDataModule

def plot_1d_burgers(data, path):
    plt.plot(data)
    plt.savefig(path)
    plt.clf()

def plot_1d_burgers2(x, y, pred, t, path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    # Plot data on each subplot
    ax1.plot(x, label='x')
    ax2.plot(y, label='y')
    ax3.plot(pred, label='pred')

    # Set titles and labels for each subplot
    ax1.set_title('Initial condition (t=0)')
    ax2.set_title(f'Soln at t={t}')
    ax3.set_title(f'Pred at t={t}')
    ax1.set_xlabel('x')
    ax2.set_xlabel('x')
    ax3.set_xlabel('x')
    ax1.set_ylabel('u')
    ax2.set_ylabel('u')
    ax3.set_ylabel('u')

    # Add legend to each subplot
    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.savefig(path)
    plt.clf()

data = PDEDataModule(
    num_examples=8196,
    num_test_examples=1024,
    num_initial_conditions=20,
    batch_size=16,
    data_dir="/usr/xtmp/jwl50/PDEBench/data/",
    file_name="1D/Burgers/Train/1D_Burgers_Sols_Nu0.1.hdf5",
    pde="1d_burgers_icl_trange"
)
data.setup()
data = data.dataset

ckpt_paths = []

# FNO
ckpt_path = "/usr/xtmp/jwl50/safari/outputs/2023-04-24/08-18-36-321937/long-conv/3k5gajow/checkpoints/epoch=2405-step=1234278 copy.ckpt"
ckpt_paths.append(ckpt_path)
# Hyena + FNO
ckpt_path = "/usr/xtmp/jwl50/safari/outputs/2023-04-14/13-16-22-864917/long-conv/gq6515ra/checkpoints/epoch=4999-step=2565000.ckpt"
ckpt_paths.append(ckpt_path)

for ckpt_num, ckpt_path in enumerate(ckpt_paths):

    model = SequenceLightningModule.load_from_checkpoint(ckpt_path, map_location="cuda")
    net = model.model.to(torch.device("cuda:0"))
    net = net.eval()
    # data = model.dataset.dataset
    print("Loaded model")

    prev_train_x = None
    prev_train_y = None
    # get one train example
    for ex_num in range(25):
        # print(ex_num)
        train_x, train_y, train_t = data["train"][ex_num] # 1024 x 1
        # print(train_x.shape)
        # print(train_y.shape)
        # print(prev_train_x == train_x)
        # print(prev_train_y == train_y)
        prev_train_x = train_x
        prev_train_y = train_y
        # plot_1d_burgers(train_x.numpy(), "hyena_train_x.png")
        # plot_1d_burgers(train_y.numpy(), "hyena_train_y.png")
        # plot_1d_burgers(train_x[-1].numpy(), "hyena_train_x.png")
        # plot_1d_burgers(train_y[-1].numpy(), "hyena_train_y.png")
        with torch.no_grad():
            # train_pred = net(train_x.unsqueeze(0)).squeeze(0)
            if ckpt_num == 0: # FNO
                train_pred = net(train_x.unsqueeze(0).cuda(), train_t.unsqueeze(0).cuda())[0]
            else:
                train_pred = net(train_x.unsqueeze(0).cuda())[0]
            # print(train_pred.shape)
            train_pred = train_pred.squeeze(0)
        # plot_1d_burgers(train_pred.numpy(), "hyena_train_pred.png")
        # plot_1d_burgers(train_pred[-1].numpy(), "hyena_train_pred.png")
        # plot_1d_burgers2(train_x.numpy(), train_y.numpy(), train_pred.numpy(), f"results/hyena_train_{ex_num}.png")
        plot_1d_burgers2(train_x[-1].cpu().numpy(), train_y[-1].cpu().numpy(), train_pred[-1].cpu().numpy(), train_t.cpu().numpy(), f"results/hyena_{ckpt_num}_train_{ex_num}.png")

        # get one test example
        test_x, test_y, test_t = data["test"][ex_num] # 1024 x 1
        # plot_1d_burgers(test_x.numpy(), "hyena_test_x.png")
        # plot_1d_burgers(test_y.numpy(), "hyena_test_y.png")
        # plot_1d_burgers(test_x[-1].numpy(), "hyena_test_x.png")
        # plot_1d_burgers(test_y[-1].numpy(), "hyena_test_y.png")
        with torch.no_grad():
            # test_pred = net(test_x.unsqueeze(0)).squeeze(0)
            if ckpt_num == 0: # FNO
                test_pred = net(test_x.unsqueeze(0).cuda(), test_t.unsqueeze(0).cuda())[0].squeeze(0)
            else:
                test_pred = net(test_x.unsqueeze(0).cuda())[0].squeeze(0)
        # plot_1d_burgers(test_pred.numpy(), "hyena_test_pred.png")
        # plot_1d_burgers(test_pred[-1].numpy(), "hyena_test_pred.png")
        # plot_1d_burgers2(test_x.numpy(), test_y.numpy(), test_pred.numpy(), f"results/hyena_test_{ex_num}.png")
        plot_1d_burgers2(test_x[-1].cpu().numpy(), test_y[-1].cpu().numpy(), test_pred[-1].cpu().numpy(), test_t.cpu().numpy(), f"results/hyena_{ckpt_num}_test_{ex_num}.png")
