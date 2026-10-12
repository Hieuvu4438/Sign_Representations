from abc import ABC, abstractmethod


class BackboneAdapter(ABC):
    @abstractmethod
    def load(self, registry_entry, device):
        """Verify explicit local assets and load their declared architecture."""

    @abstractmethod
    def extract(self, frames, timestamps, rois):
        """Return a validated FeatureBundle using the declared supports."""

    @abstractmethod
    def provenance(self) -> dict:
        pass
