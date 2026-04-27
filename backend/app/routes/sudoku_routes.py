from flask import Blueprint, request, jsonify
from app.services.solver import solve_sudoku
from app.utils.helpers import is_valid_sudoku_board

sudoku_bp = Blueprint("sudoku", __name__)

@sudoku_bp.route("/solve", methods=["POST"])
def solve():
    """
    Solves a Sudoku puzzle.
    Expects JSON: {"board": [[...], ...]}
    """
    data = request.get_json(silent=True)

    if data is None or "board" not in data:
        return jsonify({"error": "Invalid JSON or missing 'board' key"}), 400

    board = data["board"]

    if not is_valid_sudoku_board(board):
        return jsonify({"error": "Board must be a 9x9 grid with values 0-9"}), 400

    # Note: solve_sudoku modifies the board in-place
    solution = solve_sudoku(board)

    if solution is None:
        return jsonify({
            "status": "error",
            "message": "The provided Sudoku puzzle has no valid solution."
        }), 422  # Unprocessable Entity is more appropriate for valid syntax but invalid logic

    return jsonify({
        "status": "success",
        "solution": solution
    })
