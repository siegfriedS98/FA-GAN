import os
import shutil

import torch
from cv2.cv2 import imwrite
from sklearn.metrics import confusion_matrix, accuracy_score
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.types import Device
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from common import log
from ex3.constants import USE_SIGMOID

logger = log.get_logger("ex3-util")


def noise(data_size, noise_features: int = 100, range_0_1: bool = False):
    """
    Generates data_size number of random noise
    """
    n = torch.rand(data_size, noise_features) if range_0_1 else torch.randn(data_size, noise_features)
    return n


def im_convert(tensor):
    """
        Convert Tensor to displayable format
    """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy()[0, :, :]
    if not USE_SIGMOID:
        image = image * 0.5 + 0.5
        pass

    image = image.clip(0, 1)

    return image


def cnn_train(
        net: nn.Module,
        data_loader: DataLoader,
        test_data_loader: DataLoader,
        optimizer: Optimizer,
        epochs: int,
        device: Device,
        criterion=F.cross_entropy,
):
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        dataset_size = 0
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            dataset_size += len(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            pass

        logger.debug(
            f'Epoch {epoch + 1}: loss: %.3f' % (running_loss / dataset_size)
        )
        pass

        cnn_test(
            net=net,
            data_loader=test_data_loader,
            device=device,
        )
    pass


@torch.no_grad()
def cnn_test(
        net: nn.Module,
        data_loader: DataLoader,
        device: Device
):
    all_predictions = torch.tensor([])
    all_labels = torch.tensor([])
    for batch in data_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        predictions = net(images).argmax(dim=1)
        all_predictions = torch.cat(
            (all_predictions, predictions),
            dim=0
        )
        all_labels = torch.cat(
            (all_labels, labels),
            dim=0
        )
        pass

    cm = confusion_matrix(
        y_true=all_labels,
        y_pred=all_predictions
    )
    accuracy = accuracy_score(
        y_true=all_labels,
        y_pred=all_predictions
    )

    logger.debug(
        f"Accuracy: {accuracy}\n"
        "Confusion Matrix:\n"
        f"{cm}"
    )
    pass


def dataset_to_pngs(dataset: Dataset, foldername: str):
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, "data", foldername)

    logger.info(f"Extracting images to {data_dir}")
    logger.debug("Deleting old PNGs if they existed")

    for label in range(0, 10):
        label_dir = os.path.join(data_dir, f"{label}")
        if os.path.exists(label_dir):
            shutil.rmtree(label_dir)
            pass
        pass

    for (index, sample) in enumerate(dataset):
        image = sample[0].numpy()[0, :, :] * 255
        label = sample[1].numpy().argmax() if isinstance(sample[1], Tensor) else sample[1]

        label_dir = os.path.join(data_dir, f"{label}")
        os.makedirs(label_dir, exist_ok=True)

        filename = os.path.join(label_dir, "%05d.png" % index)

        imwrite(
            filename=filename,
            img=image
        )
        pass
    pass


def read_synthetic_data(filepath: str):
    return torch.load(filepath)