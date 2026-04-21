import torch

from helper_models.ResidualBlock import ResidualBlock
from helper_models.ResidualNetwork import ResNet


class TestResidualBlock:
    def test_output_shape_same_channels(self):
        block = ResidualBlock(16, 16)
        x = torch.randn(2, 16, 8, 8)
        assert block(x).shape == (2, 16, 8, 8)

    def test_output_shape_different_channels(self):
        block = ResidualBlock(16, 32)
        x = torch.randn(2, 16, 8, 8)
        assert block(x).shape == (2, 32, 8, 8)

    def test_shortcut_projection_when_channels_differ(self):
        block = ResidualBlock(16, 32)
        assert len(block.shortcut) == 2  # Conv2d + BatchNorm2d

    def test_shortcut_identity_when_channels_match(self):
        block = ResidualBlock(16, 16)
        assert len(block.shortcut) == 0

    def test_stride_downsamples_spatial(self):
        block = ResidualBlock(16, 16, stride=2)
        x = torch.randn(2, 16, 8, 8)
        assert block(x).shape == (2, 16, 4, 4)

    def test_residual_connection_changes_output(self):
        block = ResidualBlock(16, 16)
        block.eval()
        x = torch.randn(1, 16, 8, 8)
        # Zero out shortcut to isolate; with residual the output differs from conv-only
        out = block(x)
        conv_only = block.relu(
            block.bn2(block.conv2(block.relu(block.bn1(block.conv1(x)))))
        )
        assert not torch.allclose(out, conv_only)


class TestResNet:
    def test_output_shape_no_head(self):
        net = ResNet(3, [2, 2], [64, 32])
        x = torch.randn(2, 3, 16, 16)
        assert net(x).shape == (2, 32, 16, 16)

    def test_output_shape_with_classification_head(self):
        net = ResNet(3, [2, 2], [64, 32], num_classes=10)
        x = torch.randn(2, 3, 16, 16)
        assert net(x).shape == (2, 10)

    def test_custom_kernel_size(self):
        net = ResNet(3, [2], [32], kernel_size=5)
        x = torch.randn(2, 3, 16, 16)
        assert net(x).shape == (2, 32, 16, 16)

    def test_single_layer_config(self):
        net = ResNet(1, [1], [16])
        x = torch.randn(2, 1, 8, 8)
        assert net(x).shape == (2, 16, 8, 8)
