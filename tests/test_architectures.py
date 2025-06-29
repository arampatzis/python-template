"""
Test the template architectures.

This module contains tests for the template architectures.
"""

from template.trainer import FeedForward


def test_ff() -> None:
    """
    Test the feed-forward network.

    Verifies that the final output of the feed-forward network
    is a single neuron.
    """
    model = FeedForward(10, 10)

    assert model.fc[-1].out_features == 1
