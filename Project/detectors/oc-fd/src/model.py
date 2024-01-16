# Import the standard libraries
import os, glob, re, csv, time, datetime, argparse
# Import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# Import skimage
from skimage import io, transform
# Import numpy
import numpy as np
# Import tqdm
from tqdm import tqdm
# Import collections
from collections import Counter
# Import sklearn
from sklearn import metrics
# Import scipy
from scipy.interpolate import interp1d

# Import pytorch libraries
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.nn.modules.module import _addindent
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau



####################################################################################################
# for evaluation/testing
def mse_loss_cal(input, target, avg_batch=True):
    ret = torch.mean((input - target) ** 2)
    return ret.item() 


####################################################################################################
class VAE_CNN(nn.Module):
    def __init__(self):
        super(VAE_CNN, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        #self.drop = nn.Dropout(0.2)

        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(25 * 25 * 16, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc21 = nn.Linear(1024, 1024)
        self.fc22 = nn.Linear(1024, 1024)

        # Sampling vector
        self.fc3 = nn.Linear(1024, 1024)
        self.fc_bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 25 * 25 * 16)
        self.fc_bn4 = nn.BatchNorm1d(25 * 25 * 16)

        self.relu = nn.ReLU()

        # Decoder

        self.conv5 = nn.ConvTranspose2d(
            16, 64, kernel_size=3, stride=1, padding=1,   bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=1, padding=1,   bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(
            16, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def encode(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3)))
        conv4 = conv4.view(-1, 25 * 25 * 16)

        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        return r1, r2

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.50).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))
        fc4 = fc4.view(-1, 16, 25, 25)

        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        conv8 = self.conv8(conv7)
        return conv8.view(-1, 3, 100, 100)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss used for training the model
    # Reconstruction + KL divergence losses summed over all elements and batch
    # Penalize the model for not being able to reconstruct the input
class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD

# Loss used for inference
    # Used only MSER between input and output
    # No KL divergence
class inferenceLoss(nn.Module):
    def __init__(self):
        super(inferenceLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)

        return loss_MSE


###############################################################################################################
############################################### MODEL CONTROLER ###############################################
###############################################################################################################

class ModelControler:
    def __init__(self, train_root, val_root, test_root, weights_path):
        # Data loader parameters
        if train_root is None:
            # Only use the real images for training
            self.train_root = '/datasets/face-forensics/train/real'
        else:
            self.train_root = train_root

        if val_root is None:
            self.val_root = '/datasets/face-forensics/val/'
        else:
            self.val_root = val_root

        if test_root is None:
            self.test_root = '/datasets/face-forensics/test/'
        else:
            self.test_root = test_root

        if weights_path is not None:
            # Find most recent weights that are saved in the output directory
            self.weights_path = weights_path
            self.weights_flag = True
        else:
            self.weights_flag = False

        # Training parameters
        self.batch_size = 1024
        self.epochs = 200
        self.logging_interval = 50
        self.lr_initial = 1e-3

        # Output directory
        self.output_dir = '/output/'
        
        # Set torch
        torch.manual_seed(1)


    def train_validate_model(self):
        # Setup cuda
        no_cuda = False
        cuda = not no_cuda and torch.cuda.is_available()

        device = torch.device("cuda" if cuda else "cpu")
        print("Device: ", device)

        kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

        TRANSFORM_IMG = transforms.Compose([
        transforms.Resize((100, 100)),
        # transforms.CenterCrop(100),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        TRANSFORM_IMG_TEST = transforms.Compose([
            transforms.Resize((100, 100)),
            # transforms.CenterCrop(100),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Load the datasets
        print("Loading the datasets...")
        train_dataset = datasets.ImageFolder(self.train_root, transform=TRANSFORM_IMG)
        val_dataset = datasets.ImageFolder(self.val_root, transform=TRANSFORM_IMG_TEST)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, shuffle=False)

        # Create the model
        print("Creating the model...")
        model = VAE_CNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr_initial)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', threshold_mode='rel', factor=0.1, patience=20, threshold=0.01, cooldown=0, eps=1e-5,verbose=True)

        # Loss function
        loss_mse = customLoss()

        # For storing performance metrics in training
        train_losses = []
        val_losses_df = []
        val_losses_r = []
        eers = []
        aucs = []

        # Print model and dataset stats
        print('\nModel summary:')
        print(summary(model, (3, 100, 100)))

        print('\nDataset summary:')
        print("Train dataset:", train_dataset.classes)
        print(dict(Counter(train_dataset.targets)))
        print("Validation dataset:", val_dataset.classes)
        print(dict(Counter(val_dataset.targets)))

        # Create output directory
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = 'FF_split'
        output_time_dir = f'output_train_{current_datetime}_{dataset_name}'
        os.makedirs(self.output_dir + output_time_dir, exist_ok=True)

        log_filename = f'log_{dataset_name}_{current_datetime}.txt'

        out_string = f"Training. Using {self.train_root} for training and {self.val_root} for validation."
        print(out_string)
        with open(os.path.join(self.output_dir + output_time_dir, log_filename), 'a') as file:
            file.write(out_string)
            file.write(os.linesep)

        start_time = time.time()

        # Training loop
        for epoch in range(1, self.epochs + 1):
            # Epoch setup
            epoch_start_time = time.time()
            model.train()
            train_loss = 0
            
            # TRAIN
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
        
                #permute = [2, 1, 0]
                #data = data[:, permute, :, :]
        
                recon_batch, mu, logvar = model(data)
        
                loss = loss_mse(recon_batch, data, mu, logvar)
        
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

                # Logging
                if batch_idx % self.logging_interval == 0:
                    out_string = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        batch_idx * len(data) * 100 / len(train_loader.dataset),
                        loss.item() / len(data))

                    # Write to console and logfile
                    print(out_string)
                    with open(os.path.join(self.output_dir + output_time_dir, log_filename), 'a') as file:
                        file.write(out_string)
                        file.write(os.linesep)

            epoch_elapsed_time = datetime.timedelta(seconds=(time.time() - epoch_start_time))
        
            out_string = '====> Epoch: {} Average loss: {:.4f}, learning rate: {}, time per epoch: {}'.format(epoch, train_loss / len(train_loader.dataset), optimizer.state_dict()['param_groups'][0]['lr'], epoch_elapsed_time)
            print(out_string)
            with open(os.path.join(self.output_dir + output_time_dir, log_filename), 'a') as file:
                file.write(out_string)
                file.write(os.linesep)
            train_losses.append(train_loss / len(train_loader.dataset))


            # VALIDATION
            model.eval()

            # List of losses for single validation run 
            val_losses_singlerun = []
            val_losses_df_singlerun = [] 
            val_losses_r_singlerun = []
            val_labels_singlerun = []


            with torch.no_grad():
                for data, label in val_loader:
                
                    data = data.to(device)
                    recon_score, mu, logvar = model(data)
                    loss = loss_mse(recon_score, data, mu, logvar) # Reconstruction loss
                    val_losses_singlerun.append(loss.item())
                    val_labels_singlerun.append(label.item())

                    # Separate scores by class. I refuse to do the arrayal gymnastics.
                    if label.item() == 0:
                        val_losses_df_singlerun.append(loss.item())
                    else:
                        val_losses_r_singlerun.append(loss.item())

            # Validation loss
            val_loss_df = np.mean(val_losses_df_singlerun)
            val_loss_r = np.mean(val_losses_r_singlerun)
            val_losses_df.append(val_loss_df)
            val_losses_r.append(val_loss_r)
            # AUC
            # Deepfakes vs real images
            fpr, tpr, thresholds = metrics.roc_curve(val_labels_singlerun, val_losses_singlerun, pos_label=0) 
            roc_auc = metrics.auc(fpr, tpr)
            aucs.append(roc_auc)
            # EER
            fnr = 1 - tpr
            eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
            eers.append(eer)
            # BPCER, APCER
            fpr_at_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
            fnr_at_eer = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

            # Print validation results
            out_string = f'====> Validation: deepfake images loss: {val_loss_df:.3f}, real images loss: {val_loss_r:.3f}, EER: {eer:.3f}, AUC: {roc_auc:.3f}, BPCER(FPR)@EER: {fpr_at_eer:.3f}, APCER(FNR)@EER: {fnr_at_eer:.3f}'
            print(out_string)
            with open(os.path.join(self.output_dir + output_time_dir, log_filename), 'a') as file:
                file.write(out_string)
                file.write(os.linesep)

            scheduler.step(val_loss_bf)
            print(f'Number of bad epochs: {scheduler.num_bad_epochs}')

            # If scheduler reached the lr limit and there are too many bad epochs, early stop the training.
            if (scheduler.num_bad_epochs >= scheduler.patience) and (optimizer.state_dict()['param_groups'][0]['lr'] * scheduler.factor < scheduler.eps):
                out_string = f'Early stopping.'
                print(out_string)
                with open(os.path.join(output_root_dir, log_filename), 'a') as file:
                    file.write(out_string)
                    file.write(os.linesep)
                break

            # Save the weights every 10-th epoch
            if (epoch % 10) == 0:
                current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'weights_autosave_{epoch}_train-{train_losses[-1]:.3f}_val-{val_losses_df[-1]:.3f}_eer-{eers[-1]:.3f}_{current_datetime}.pth'
                torch.save(model.state_dict(), os.path.join(self.output_dir + output_time_dir, filename))


        print('\n----------------------------------------------------------------')
        # Print total training time
        end_time = time.time()
        elapsed_time = datetime.timedelta(seconds=(end_time - start_time))
        print(f"Total training time: {elapsed_time}")

        # Save the trained weights
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        weights_filename = f'weights_final_train-{train_losses[-1]:.3f}_val-{val_losses_df[-1]:.3f}_eer-{eers[-1]:.3f}_{current_datetime}.pth'
        torch.save(model.state_dict(), os.path.join(self.output_dir + output_time_dir, weights_filename))

        # Plot loss over time
        plt.figure(figsize=(15, 10))
        plt.plot(range(len(train_losses[1:])), train_losses[1:], c="dodgerblue")
        plt.plot(range(len(val_losses_df[1:])), val_losses_df[1:], c="r")
        plt.title("Loss per epoch", fontsize=18)
        plt.xlabel("epoch", fontsize=18)
        plt.ylabel("loss", fontsize=18)
        plt.legend(['Training loss', 'Validation loss'], fontsize=18)
        filename = f'loss.svg'
        plt.savefig( os.path.join(self.output_dir + output_time_dir, filename))
        
        # Plot equal error rate over time
        plt.figure(figsize=(15, 10))
        plt.plot(range(len(eers[1:])), eers[1:], c="dodgerblue")
        plt.plot(range(len(aucs[1:])), aucs[1:], c="r")
        plt.title("EER and AUC per epoch", fontsize=18)
        plt.xlabel("epoch", fontsize=18)
        # plt.ylabel("EER, AUC", fontsize=18)
        plt.legend(['EER', 'AUC'], fontsize=18)
        filename = f'eer-auc.svg'
        plt.savefig( os.path.join(self.output_dir + output_time_dir, filename))

        # Plot df vs real loss over time 
        plt.figure(figsize=(15, 10))
        plt.plot(range(len(val_losses_df[1:])), val_losses_df[1:], c="r")
        plt.plot(range(len(val_losses_r[1:])), val_losses_r[1:], c="dodgerblue")
        plt.title("Reconstruction score during validation", fontsize=18)
        plt.xlabel("epoch", fontsize=18)
        plt.ylabel("Reconstruction score", fontsize=18)
        plt.legend(['Deepfake images reconstruction score', 'Real images reconstruction score'], fontsize=18)
        filename = f'reconstruction_score.svg'
        plt.savefig( os.path.join(self.output_dir + output_time_dir, filename))

        if not self.weights_flag:
            self.weights_path = os.path.join(self.output_dir + output_time_dir, weights_filename)

        return 

    # Test model
    def test_model(self):
        no_cuda = False
        cuda = not no_cuda and torch.cuda.is_available()
        
        device = torch.device("cuda" if cuda else "cpu")
        print("Device:", device)
        
        kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
        
        # Possible image transformations
        # TRANSFORM_IMG = transforms.Compose([
        #     transforms.Resize((100,100)),
        #     # transforms.CenterCrop(100),
        #     transforms.RandomHorizontalFlip(),
        #     # transforms.RandomVerticalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # ])
        TRANSFORM_IMG_TEST = transforms.Compose([
            transforms.Resize((100,100)),
            # transforms.CenterCrop(100),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Load dataset
        print('Loading the dataset...')
        test_dataset = datasets.ImageFolder(self.test_root, transform=TRANSFORM_IMG_TEST)
        test_loader = DataLoader(test_dataset, shuffle=False)

        # Create model
        model = VAE_CNN().to(device)

        # Load weights
        ckpt = torch.load(self.weights_path)
        model.load_state_dict(ckpt)
        model = model.to(device)

        print("Begin testing...")
        start_time = time.time()

        scores = []
        labels = []

        scores_df = []
        scores_r = []

        dataset_name = self.test_root.split(os.sep)[-1]
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f'oc-fd'
        output_time_dir = f'output_test_{current_datetime}_{dataset_name}_{model_name}'
        os.makedirs(self.output_dir + output_time_dir, exist_ok=True)

        out_string = f"Testing. Using {self.test_root} for testing and using {self.weights_path} for model weights."
        log_filename = f'log_{dataset_name}_{current_datetime}.txt'
        print(out_string)
        with open(os.path.join(self.output_dir + output_time_dir, log_filename), 'a') as file:
            file.write(out_string)
            file.write(os.linesep)

        model.eval()

        with torch.no_grad():
            for data, label in tqdm(test_loader, desc="Testing..."):
                data = data.to(device)
                reconstruction, mu, logvar = model(data)
                loss = mse_loss_cal(reconstruction, data)
                scores.append(loss)
                labels.append(label.item())

                # Separate scores by class
                if label.item() == 0:
                    scores_df.append(loss)
                else:
                    scores_r.append(loss)

        # Print total testing time
        end_time = time.time()
        elapsed_time = datetime.timedelta(seconds=(end_time - start_time))
        print(f"Total testing time: {elapsed_time}")

        # Get ROC curve and AUC
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
        roc_auc = metrics.auc(fpr, tpr)
        # Save the ROC curve to output directory
        plt.figure(figsize=(15, 10))
        plt.plot(fpr, tpr, c="dodgerblue")
        plt.title("ROC curve", fontsize=18)
        plt.xlabel("FPR", fontsize=18)
        plt.ylabel("TPR", fontsize=18)
        filename = f'ROC_{dataset_name}_{current_datetime}.svg'
        plt.savefig( os.path.join(self.output_dir + output_time_dir, filename))
        print(f"AUC: {roc_auc}")

        ## Calculate EER, treshold@EER, APCER@EER and BPCER@EER
        # APCER - proportion of deepfakes incorrectly classified as real images - false negative rate
        # BPCER - proportion of real images incorrectly classified as deepfakes - false positive rate
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        fpr_at_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        fnr_at_eer = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

        # Store the labels and scores in a csv file
        with open(os.path.join(self.output_dir + output_time_dir, 'labels.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(labels)
        with open(os.path.join(self.output_dir + output_time_dir, 'scores.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(scores)

        # Log the output
        out_string = f'EER: {eer:.4f}, EER treshold: {eer_threshold:.4f}, BPCER(FPR)@EER: {fpr_at_eer:.4f}, APCER(FNR)@EER: {fnr_at_eer:.4f}'
        print(out_string)
        with open(os.path.join(self.output_dir + output_time_dir, log_filename), 'a') as file:
            file.write(out_string)
            file.write(os.linesep)


        # Plot the distribution 
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        min_score = min(min(scores_df), min(scores_r))
        max_score = max(max(scores_df), max(scores_r))
        axs[0].hist(scores_df, bins=30, color='blue', alpha=0.7)
        axs[0].set_title('Reconstruction score distribution')
        axs[0].axvline(eer_threshold, color='red', linestyle='dashed', linewidth=2)  # Draw treshold
        axs[0].set_xlim(min_score, max_score)
        axs[0].set_ylabel('Frequency')
        axs[1].hist(scores_r, bins=30, color='green', alpha=0.7)
        axs[1].axvline(eer_threshold, color='red', linestyle='dashed', linewidth=2)  # Draw treshold
        axs[1].set_xlim(min_score, max_score)
        axs[1].set_xlabel('Reconstruction score')
        axs[1].set_ylabel('Frequency')

        # Add legend
        real_patch = mpatches.Patch(color='blue', label='Deepfake')
        df_patch = mpatches.Patch(color='green', label='Real')
        threshold_line = plt.Line2D([0], [0], color='red', linestyle='dashed', linewidth=2, label='Threshold')
        plt.legend(handles=[real_patch, df_patch, threshold_line], loc='upper right')
        
        plt.tight_layout()
        filename = f'distribution_{dataset_name}_{current_datetime}.svg'
        plt.savefig(os.path.join(self.output_dir + output_time_dir, filename))

        # Calculate percission, recall and F1 score
        precision, recall, thresholds_pr = metrics.precision_recall_curve(labels, scores, pos_label=0)
        pr_auc = metrics.auc(recall, precision)

        # Calculate F1 at threshold
        f1_scores = [2 * (p * r) / (p + r + 1e-10) for p, r in zip(precision, recall)]
        f1_max_idx = np.argmax(f1_scores)
        f1_max_threshold = thresholds_pr[f1_max_idx]

        # Log the output
        out_string = f'PR AUC: {pr_auc:.4f}, F1 score: {f1_scores[f1_max_idx]:.4f}, F1 threshold: {f1_max_threshold:.4f}'
        print(out_string)
        with open(os.path.join(self.output_dir + output_time_dir, log_filename), 'a') as file:
            file.write(out_string)
            file.write(os.linesep)

        # Plot Precision-Recall curve
        plt.figure(figsize=(15, 10))
        plt.plot(recall, precision, color='blue', lw=2, label='PR curve (AUC = {:.2f})'.format(pr_auc))
        plt.scatter(recall[f1_max_idx], precision[f1_max_idx], marker='o', color='red', label='F1 point ({:.4f}, {:.4f})'.format(recall[f1_max_idx], precision[f1_max_idx]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall (PR) Curve')
        plt.legend(loc="lower left")
        filename = f'PR_{dataset_name}_{current_datetime}.svg'
        plt.savefig(os.path.join(self.output_dir + output_time_dir, filename))

        # Generate classification report
        y_pred = (np.array(scores) > f1_max_threshold).astype(int)
        target_names = ['Deepfake', 'Real']
        classification_rep = metrics.classification_report(labels, y_pred, target_names=target_names)
        print(classification_rep)
        with open(os.path.join(self.output_dir + output_time_dir, log_filename), 'a') as file:
            file.write(classification_rep)
            file.write(os.linesep)

        # Save the model
        torch.save(model.state_dict(), os.path.join(self.output_dir + output_time_dir, 'model.pth'))

        return

####################################################################################################
############################################### MAIN ###############################################
####################################################################################################

if __name__ == '__main__':

    # Parser arguments
    parser = argparse.ArgumentParser(description="Script to train or test the OC-FakeDect model.")

    parser.add_argument('--train', action='store_true', help='Train the model.')
    parser.add_argument('--test', action='store_true', help="Test the model.")
    parser.add_argument('--traindata', type=str, help='Location of the training dataset. Used for training. Contains only real class.')
    parser.add_argument('--valdata', type=str, help="Location of the validation dataset. Used when training. Contains real and deepfake classes.")

    parser.add_argument('--testdata', type=str, help="Path to test dataset, containing real and deepfake classes.")
    parser.add_argument('--weights', type=str, help="Path to model weights. Loaded when testing the model.")

    # Parse the arguments
    args = parser.parse_args()

    # Create ModelEvaluator class
    modelControler = ModelControler(args.traindata, args.valdata, args.testdata, args.weights)

    # If no arguments or --help is provided, the help message will be displayed
    if not args.train and not args.test:
        parser.print_help()
    else:
        if args.train:
            modelControler.train_validate_model()
        if args.test:
            modelControler.test_model()