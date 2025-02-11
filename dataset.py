from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms


def generate_dataset():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder('dataset', transform=transform)
    # print(dataset.classes)
    # print(dataset.class_to_idx)

    train_size = round(.8 * len(dataset))
    test_size = round(.15 * len(dataset))
    validation_size = len(dataset) - (train_size + test_size)

    train_dataset, test_dataset, validation_dataset = random_split(dataset, [train_size, test_size, validation_size])

    train_dataset = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = DataLoader(test_dataset, batch_size=32, shuffle=False)
    validation_dataset = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    return train_dataset, test_dataset, validation_dataset
