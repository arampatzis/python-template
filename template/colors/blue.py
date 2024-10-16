"""Blue module is used to demonstrate the documentation capabilities."""
from abc import ABC, abstractmethod

from template.square import A

# ruff: noqa


class C(ABC):
    """Dummy class C."""

    def __init__(self):
        """Class constructor."""
        print("C")

    @abstractmethod
    def c(self):
        """Abstract method."""
        ...


class D(C, A):
    """Dummy class D."""

    def __init__(self):
        """Class constructor."""
        super().__init__()

    def a(
        self,
    ):
        """Implementation of the abstract method a."""

    def c(
        self,
    ):
        """Implementation of the abstract method c."""
