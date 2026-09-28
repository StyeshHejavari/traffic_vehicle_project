import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(
        self,
        num_classes=8,
        dropout=0.0,
        use_gap=False
    ):
        super().__init__()

        self.use_gap = use_gap

        # =========================
        # Feature Extractor
        # =========================

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        # =========================
        # Classifier
        # =========================

        if use_gap:

            self.pool = nn.AdaptiveAvgPool2d(
                (1, 1)
            )

            classifier_input = 64

        else:

            self.pool = nn.Identity()

            classifier_input = 64 * 56 * 56

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                classifier_input,
                128
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                128,
                num_classes
            )
        )

    # =========================
    # Forward
    # =========================

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x



class SimpleCNNAvgPool(nn.Module):

    def __init__(
        self,
        num_classes=8,
        dropout=0.0,
        use_gap=False
    ):
        super().__init__()

        self.use_gap = use_gap

        # =========================
        # Feature Extractor
        # =========================

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AvgPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AvgPool2d(2)
        )

        # =========================
        # Classifier
        # =========================

        if use_gap:

            self.pool = nn.AdaptiveAvgPool2d(
                (1, 1)
            )

            classifier_input = 64

        else:

            self.pool = nn.Identity()

            classifier_input = 64 * 56 * 56

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                classifier_input,
                128
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                128,
                num_classes
            )
        )

    # =========================
    # Forward
    # =========================

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x