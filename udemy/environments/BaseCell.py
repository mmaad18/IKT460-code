from udemy.environments.CellType import CellType


class BaseCell:
    def __init__(self, cell_type: CellType):
        self.cell_type = cell_type
