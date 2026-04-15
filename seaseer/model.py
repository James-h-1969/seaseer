import torch
import torch.nn as nn

from helper_models.ResidualNetwork import ResNet


class SeaSeer(nn.Module):
    """SeaSeer: Neural ODE-based spatiotemporal forecasting model.

    Uses a velocity network to learn transport dynamics and an
    optional emission network to estimate prediction uncertainty.

    Args:
        in_channels: Number of input state variables.
        out_channels: Number of output state variables to predict.
        hidden_channels: Channel sizes for residual layers.
        num_blocks: Block counts per residual layer.
        use_attention: Whether to use an attention-based velocity branch.
        use_emission: Whether to include an emission (noise) model.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        num_blocks=None,
        use_attention=False,
        use_emission=False,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [128, 64, 2 * out_channels]
        if num_blocks is None:
            num_blocks = [5, 3, 2]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.use_emission = use_emission

        # velocity network (f_conv)
        self.velocity_net = ResNet(in_channels, num_blocks, hidden_channels)

        # optional attention velocity branch (f_att)
        if use_attention:
            self.attention_net = ResNet(in_channels, num_blocks, hidden_channels)
            self.gamma = nn.Parameter(torch.tensor(0.1))

        # optional emission model for uncertainty estimation
        if use_emission:
            self.emission_net = ResNet(
                in_channels + out_channels,
                [3, 2, 2],
                [128, 64, 2 * out_channels],
            )

    def compute_velocity(self, x):
        """Compute the velocity field from input state.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Velocity tensor of shape (B, 2*out_channels, H, W).
        """
        v = self.velocity_net(x)
        if self.use_attention:
            v = v + self.gamma * self.attention_net(x)
        return v

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Prediction tensor of shape (B, out_channels, H, W).
            If use_emission, returns (mean, std) tuple instead.
        """
        velocity = self.compute_velocity(x)

        # TODO: integrate ODE dynamics here
        # For now, pass through velocity net as a baseline
        out = velocity[:, : self.out_channels, :, :]

        if self.use_emission:
            emission_input = torch.cat([x, out], dim=1)
            emission = self.emission_net(emission_input)
            mean = out + emission[:, : self.out_channels, :, :]
            std = nn.functional.softplus(emission[:, self.out_channels :, :, :])
            return mean, std

        return out
