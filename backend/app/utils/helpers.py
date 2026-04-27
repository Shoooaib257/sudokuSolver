from typing import List, Any

def is_valid_sudoku_board(board: Any) -> bool:
    """
    Validate that the input is a 9x9 grid with integer values 0-9.
    """
    if not isinstance(board, list) or len(board) != 9:
        return False

    for row in board:
        if not isinstance(row, list) or len(row) != 9:
            return False
        for val in row:
            # Check if it's an integer and within the valid range
            if not isinstance(val, int) or not (0 <= val <= 9):
                return False

    return True
