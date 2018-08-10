import torch
from torch import nn

class AlexNet_v1(nn.Module):
  """AlexNet version 1
  Described in: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

  Args:
    num_classes: number of predicted classes
  """

  def __init__(self, num_classes):
    super(AlexNet_v1, self).__init__()

    self.features = nn.Sequential(
      nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(96, 256, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(256, 384, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(384, 384, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(384, 256, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, padding=2)
    )

    self.classifier = nn.Sequential(
      nn.Dropout(0.5)
      nn.Linear(256 * 6 * 6, 4096),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(4096, 4096),
      nn.ReLU(),
      nn.Linear(4096, num_classes)
    )

  def forward(self, x):
    """
    Args:
      x: input tensor (input shape is [batch_size, channel=3, width=224, height=224])
    """
    x = self.features(x)
    x = x.view(x.size(0), 256 * 6 * 6)
    x = self.classifier(x)
    return x

class AlexNet_v2(nn.Module):
  """AlexNet vesion 2
  Described in: https://arxiv.org/pdf/1404.5997v2.pdf

  Args:
    num_classes: number of predicted classes
  """

  def __init__(self, num_classes):
    super(AlexNet_v2, self).__init__()

    self.features = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(384, 384, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(384, 256, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2)
    )

    self.classifier = nn.Sequential(
      nn.Dropout(0.5),
      nn.Linear(256 * 6 * 6, 4096),
      nn.ReLU()
      nn.Dropout(0.5)
      nn.Linear(4096, 4096),
      nn.ReLU(),
      nn.Linear(4096, num_classes)
    )

  def forward(self, x):
    """
    Args:
      x: input tensor (input shape is [batch_size, channel=3, width=224, height=224])
    """
    x = self.features(x)
    x = x.view(x.size(0), 256 * 6 * 6)
    x = self.classifier(x)
    return x