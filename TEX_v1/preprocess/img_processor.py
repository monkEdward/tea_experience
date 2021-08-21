from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

ds_dir = './tea_experience/dataset'


def dataset_loader():
    # train and test data directory
    data_dir = "./dataset/resized_imgs"
    test_data_dir = "./dataset/resized_test_imgs"

    # load the train and test data
    # imagefolder, load img from folders with classes into folder name. imgs are been preprocessed
    dataset = ImageFolder(data_dir,
                          transform=transforms.Compose([transforms.Resize((150, 150)),
                                                        transforms.ToTensor()]))
    test_dataset = ImageFolder(test_data_dir,
                               transforms.Compose([transforms.Resize((150, 150)),
                                                   transforms.ToTensor()]))
    return dataset, test_dataset


def create_train_test_val_dataset(dataset, batch_size, validation_size):
    training_size = len(dataset) - validation_size
    train_data, val_data = random_split(dataset, [training_size, validation_size])

    # load the train and validation into batches.
    train_dl = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_data, batch_size * 2, num_workers=4, pin_memory=True)
    return train_dl, val_dl