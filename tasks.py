from enum import Enum


class Task(str, Enum):
    BOUNDARY = "boundary"
    PUNCTUATION = "punctuation"


OUTSIDE = "O"
BOUNDARY = "B"

