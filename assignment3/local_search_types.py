from enum import Enum


class LocalSearchType(Enum):
    STEEPEST = 1
    GREEDY = 2


class StartingSolutionType(Enum):
    RANDOM = 1
    GREEDY = 2


class IntraRouteMovesType(Enum):
    TWO_NODES = 1
    TWO_EDGES = 2


class MoveType(Enum):
    INTRA_TWO_NODES = 1
    INTRA_TWO_EDGES = 2
    INTER_NODES_EXCHANGE = 3
