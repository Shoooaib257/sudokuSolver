from typing import List, Optional, Tuple

# Type alias for a 9x9 Sudoku board
SudokuBoard = List[List[int]]

def is_valid(board: SudokuBoard, row: int, col: int, num: int) -> bool:
    """
    Check if placing num in board[row][col] is valid according to Sudoku rules.
    
    Args:
        board: The 9x9 Sudoku grid.
        row: Row index (0-8).
        col: Column index (0-8).
        num: Number to place (1-9).
        
    Returns:
        True if valid, False otherwise.
    """
    # Check row: 'num' must not already exist in the row
    if num in board[row]:
        return False

    # Check column: 'num' must not already exist in the column
    for r in range(9):
        if board[r][col] == num:
            return False

    # Check 3x3 subgrid: 'num' must not already exist in the box
    box_start_row, box_start_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_start_row, box_start_row + 3):
        for c in range(box_start_col, box_start_col + 3):
            if board[r][c] == num:
                return False

    return True


def find_empty_cell(board: SudokuBoard) -> Optional[Tuple[int, int]]:
    """
    Find an empty cell (represented by 0) in the Sudoku board.
    
    Returns:
        A tuple (row, col) of the first empty cell found, or None if the board is full.
    """
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return r, c
    return None


def solve_sudoku(board: SudokuBoard) -> Optional[SudokuBoard]:
    """
    Solve the Sudoku board using backtracking.
    
    The board is modified in-place.
    
    Args:
        board: The 9x9 Sudoku grid to solve.
        
    Returns:
        The solved board if a solution exists, otherwise None.
    """
    # Optimization: Scan the board for the next empty cell (0)
    # For further optimization, we could pass the last found empty cell's 
    # position to the next recursive call to avoid re-scanning from the top.
    empty_cell = find_empty_cell(board)
    
    # If no empty cell is found, the board is solved
    if not empty_cell:
        return board
    
    row, col = empty_cell

    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num

            # Recursively try to solve the rest of the board
            if solve_sudoku(board):
                return board

            # Backtrack: reset the cell if no solution is found with 'num'
            board[row][col] = 0

    return None
