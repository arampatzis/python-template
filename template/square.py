"""Square module is used to demonstrate the documentation capabilities."""
from abc import ABC, abstractmethod


# ruff: noqa


class A(ABC):
    """Dummy class A."""

    def __init__(self):
        """Class constructor."""
        print("A")

    @abstractmethod
    def a(self):
        """Abstract method."""


class B(A):
    """Dummy class B."""

    def __init__(self):
        """This is the class constructor."""
        super().__init__()

    def a(self):
        """Implementation of the abstract method."""
