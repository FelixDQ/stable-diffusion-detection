from io import BytesIO
from PIL import Image
import torch
from torchmetrics.classification import BinaryConfusionMatrix
import random
from torchvision import transforms

def tfft(x):
    return torch.stack(
                [
                    torch.fft.fft2(x[0]),
                    torch.fft.fft2(x[1]),
                    torch.fft.fft2(x[2]),
                ]
            )

def tabs(x):
    return torch.abs(x)

def tlog(x):
    return torch.log10(x)

def tnorm(x):
    return ((x - x.min()) / (x.max() - x.min()) * 2) - 1

def spec(x):
    return tnorm(tfft(x).abs().log())

def JPEGcompression(image):
    image = transforms.ToPILImage()(image)
    QUALITY_FACTOR = 85

    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=QUALITY_FACTOR)
    outputIoStream.seek(0)
    return transforms.ToTensor()(Image.open(outputIoStream))

def get_accuracy(logits, labels, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (logits.argmax(1) == labels).sum().item()
    accuracy = 100.0 * corrects/batch_size
    return accuracy

def get_confusion_matrix(logits, labels):
    ''' Obtain confusion matrix for training round '''
    preds = logits.argmax(1)
    cm = BinaryConfusionMatrix().to(get_device())
    return cm(preds, labels)


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # if not torch.backends.mps.is_available():
    #     if not torch.backends.mps.is_built():
    #         print(
    #             "MPS not available because the current PyTorch install was not "
    #             "built with MPS enabled."
    #         )
    #     else:
    #         print(
    #             "MPS not available because the current MacOS version is not 12.3+ "
    #             "and/or you do not have an MPS-enabled device on this machine."
    #         )

    # else:
    #     device = torch.device("mps")
    return device

def squeeze(bits = 4):
    round_value = 2 ** bits
    def _squeeze(x):
        return torch.round(x * round_value) / round_value
    return _squeeze

def rand_noise(x):
    return x + torch.randn_like(x) * torch.rand(1) * 0.1

def rand_pad(x):
    return transforms.Pad(padding=random.randint(0, 50))(x)