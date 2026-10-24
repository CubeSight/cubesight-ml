import abc


class Env(abc.ABC):
    @abc.abstractmethod
    def get_possible_moves(self) -> list[int]:
        pass
