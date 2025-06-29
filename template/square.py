"""Square module is used to demonstrate the documentation capabilities."""

from abc import ABC, abstractmethod


class A(ABC):
    """Dummy class A."""

    def __init__(self) -> None:
        """Initialize the class."""
        print("A")

    @abstractmethod
    def a(self) -> None:
        """Abstract method."""


class B(A):
    """Dummy class B."""

    def __init__(self) -> None:
        """Initialize the class."""
        super().__init__()

    def a(self) -> None:
        """Implement method a."""
