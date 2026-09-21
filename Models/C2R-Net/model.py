"""
C2R-Net: Coarse-to-Refined Network for Image Forgery Detection

Independent reimplementation based on:
Y. Wei, X. Bi, and B. Xiao,
"C2R Net: The Coarse to Refined Network for Image Forgery Detection,"
TrustCom/BigDataSE, 2018, pp. 1656-1659.

Note:
The original authors' complete official implementation was not used.
Some low-level implementation details are reconstructed from the
published description of the C2R-Net architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Convolution + ReLU.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class CCNN(nn.Module):
    """
    Coarse CNN.

    The C-CNN follows a VGG-16 style feature extractor and
    produces a coarse forgery probability map.

    Input:
        B x 3 x H x W

    Output:
        B x 1 x Hc x Wc
    """

    def __init__(self, num_classes=1):
        super().__init__()

        # VGG-16 style convolutional feature extractor
        self.features = nn.Sequential(

            # Block 1
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),

            # Block 2
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2),

            # Block 3
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2, 2),

            # Block 4
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d(2, 2),

            # Block 5
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
        )


        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        feat = self.features(x)
        coarse_logits = self.classifier(feat)

        return coarse_logits


class RCNN(nn.Module):
    """
    Refined CNN.

    The R-CNN follows a VGG-19 style feature extractor and
    refines the suspicious regions produced by the C-CNN.

    Input:
        image features / image

    Output:
        refined forgery probability map
    """

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            ConvBlock(in_channels, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),

            # Block 2
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2),

            # Block 3
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2, 2),

            # Block 4
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d(2, 2),

            # Block 5
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                512,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                num_classes,
                kernel_size=1
            )
        )

    def forward(self, x):
        feat = self.features(x)
        refined_logits = self.refine(feat)

        return refined_logits


class C2RNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.coarse_net = CCNN(
            num_classes=num_classes
        )

        self.refined_net = RCNN(
            in_channels=3,
            num_classes=num_classes
        )

    def forward(self, x):
        coarse_logits = self.coarse_net(x)

        # Resize coarse result to input resolution.
        coarse_logits_up = F.interpolate(
            coarse_logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        coarse_prob = torch.sigmoid(
            coarse_logits_up
        )

        refined_logits = self.refined_net(
            refined_input
        )

        refined_logits = F.interpolate(
            refined_logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        refined_prob = torch.sigmoid(
            refined_logits
        )

        return {
            "coarse_logits": coarse_logits_up,
            "coarse_prob": coarse_prob,
            "refined_logits": refined_logits,
            "refined_prob": refined_prob
        }


if __name__ == "__main__":

    model = C2RNet(num_classes=1)

    x = torch.randn(
        2, 3, 384, 256
    )

    outputs = model(x)

    for name, value in outputs.items():
        print(
            f"{name}: "
            f"{tuple(value.shape)}"
        )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nTotal parameters: "
        f"{total_params / 1e6:.2f} M"
    )
