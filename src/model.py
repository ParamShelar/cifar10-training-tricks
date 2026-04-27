import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, use_batchnorm=False, use_dropout=False, dropout_p=0.5):
        super().__init__()

        def conv_block(in_channels, out_channels):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            ]

            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))

            layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3, 32),
            nn.MaxPool2d(2),

            conv_block(32, 64),
            nn.MaxPool2d(2),

            conv_block(64, 128),
        )

        classifier_layers = [
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
        ]

        if use_dropout:
            classifier_layers.append(nn.Dropout(dropout_p))

        classifier_layers.append(nn.Linear(128, 10))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x