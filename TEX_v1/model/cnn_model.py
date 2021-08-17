import torch.nn.functional as F
import torch.nn as nn
import time as T
import torch
import os

batch_size = 64
val_size = 20

class Eta:

    def __init__(self):
        self.avg = 0.0
        self.eta = 0.0

    def update_eta(self, used_time, epoch_remaining):
        if self.avg == 0.0:
            self.avg = used_time
        else:
            self.avg = (self.avg + used_time) / 2

        self.eta = self.avg * epoch_remaining

    def show_eta(self):
        print(f'='*16, f'\nETA: {self.eta}')

class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss

        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy

        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result, start, eta, epochs):
        f_time = start - T.time()
        eta.update_eta(f_time, epochs - epoch)
        print(f"Epoch [{epoch}] processed in {f_time:.2f} mins.\n"
              f"train_loss: {result['train_loss']:.4f}, "
              f"val_loss: {result['val_loss']:.4f}, "
              f"val_acc: {result['val_acc']:.4f}")


class TeaObjectRelatedClassifierCNN(ImageClassificationBase):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Linear(82944,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,6)
        )

    def forward(self, xb):
        return self.network(xb)



def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optim = opt_func(model.parameters(), lr)
    eta = Eta()
    
    for epoch in range(epochs):
        start_time = T.time()
        model.train()
        train_losses = []

        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            optim.step()
            optim.zero_grad()

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()

        model.epoch_end(epoch, result, start_time, eta, epochs)
        history.append(result)

    return history


def model_creation(train_ds, validation_ds):
    num_epochs = 10
    opt_func = torch.optim.Adam

    #TODO reduce lr to 0.01 if computation time is low
    learning_rate = 0.1

    model = TeaObjectRelatedClassifierCNN()
    # fitting the model on training data and record the result after each epoch
    history = fit(num_epochs, learning_rate, model, train_ds, validation_ds, opt_func)

    return model, history

def save_model(model):
    # bpath = 'D:\Improve\tea_experience_project\tea_experience\TEX_v1\model\trained\'
    # os.mkdir(bpath)
    # model_path = os.path.join(bpath, 'tea_experience_model_v1.pth')
    torch.save(model.state_dict(), 'tea_experience_model_v1.pth')


def to_device(data, device):

    "Move data to the device"
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)


def predict_class(img, model, dataset):
    """ Predict the class of image and Return Predicted Class"""
    device = 'cpu'

    img = to_device(img.unsqueeze(0), device)
    prediction = model(img)
    _, preds = torch.max(prediction, dim=1)

    return dataset.classes[preds[0].item()]

def load_model():
    path = 'D:\Improve\tea_experience_project\tea_experience\TEX_v1\model\trained\tea_experience_model_v1.pth'

    model = TeaObjectRelatedClassifierCNN()
    model.load_state_dict(torch.load(path))

    return model