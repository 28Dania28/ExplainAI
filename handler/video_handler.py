from abc import ABC, abstractmethod


class VideoHandler(ABC):
    def __init__(self, url):
        self.url = url

    @abstractmethod
    def get_transcript(self):
        pass
