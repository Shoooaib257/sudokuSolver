from flask import Blueprint, request, jsonify
from app.services.solver import solve_sudoku
from app.services.image_processor import preprocess_image, find_largest_contour, get_corners, warp_perspective, split_cells, extract_board
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

@sudoku_bp.route("/scan", methods=["POST"])
def scan():
    """
    Extracts a Sudoku board from an uploaded image.
    Expects a file in the 'image' field.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files["image"]
    
    try:
        # Pipeline: Preprocess -> Find Grid -> Warp -> Split -> Extract
        thresh = preprocess_image(file)
        contour = find_largest_contour(thresh)
        corners = get_corners(contour)
        
        # We need the original color image for warping if we want to preserve color,
        # but preprocess_image already read the file. We might need to seek back or read once.
        # Let's re-read the original image from the file bytes.
        file.seek(0)
        import cv2
        import numpy as np
        file_bytes = np.frombuffer(file.read(), np.uint8)
        original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        warped = warp_perspective(original, corners)
        cells = split_cells(warped)
        board = extract_board(cells)
        
        return jsonify({
            "status": "success",
            "board": board
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
