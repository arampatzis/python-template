"""Blue module is used to demonstrate the documentation capabilities."""

from abc import ABC, abstractmethod

from template.square import A


class C(ABC):
    """Dummy class C."""

    def __init__(self) -> None:
        """Initialize the class."""
        print("C")

    @abstractmethod
    def c(self) -> None:
        """Abstract method."""
        ...


class D(C, A):
    """Dummy class D."""

    def __init__(self) -> None:
        """Initialize the class."""
        super().__init__()

    def a(
        self,
    ) -> None:
        """Implement method a."""

    def c(
        self,
    ) -> None:
        """Implement method c."""
