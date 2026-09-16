from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """The contract the rest of the system depends on.

    The model layer is an interface because clients arrive with incompatible
    constraints: one is locked into a hosted provider by an enterprise agreement,
    another cannot send data outside its own network at all, a third already runs an
    inference cluster. Supporting any of those must not require touching retrieval or
    answering - which it would if those called a vendor's client directly.

    It is a Protocol rather than a base class so a provider satisfies it by shape
    alone. An implementation wrapping someone else's SDK does not need to import
    anything from here, which is the point: the dependency runs one way.

    `runtime_checkable` allows isinstance checks, but note it verifies only that the
    methods exist, not their signatures. It is a smoke test, not a guarantee; the type
    checker is what actually enforces the contract.
    """

    def generate(self, prompt: str) -> str:
        """Return the model's answer. Raises rather than returning empty on failure."""
        ...

    def healthy(self) -> bool:
        """Whether the provider can currently serve. Never raises - readiness calls it."""
        ...
