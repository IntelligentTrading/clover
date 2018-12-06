from abc import ABC, abstractmethod

class TickListener(ABC):

    @abstractmethod
    def process_event(self, price_data, signal_data):
        pass

    @abstractmethod
    def broadcast_ended(self):
        pass