import cv2
import numpy as np

from backend.app.services.image_processor import (
    preprocess_image,
    find_largest_contour,
    get_corners,
    warp_perspective,
    split_cells,
    clean_cell,
    is_cell_empty
)

IMAGE_PATH = "data/sample_sudoku.jpg"


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image from {path}")
    return img


def main():
    # Load original image
    original = load_image(IMAGE_PATH)

    # Preprocess image
    with open(IMAGE_PATH, "rb") as f:
        thresh = preprocess_image(f)

    # Find largest contour (Sudoku grid)
    contour = find_largest_contour(thresh)

    # Draw contour
    contour_img = original.copy()
    cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 3)
    cv2.imwrite("contour_output.jpg", contour_img)

    # Get corners and warp
    corners = get_corners(contour)
    warped = warp_perspective(original, corners)

    # Save intermediate outputs
    cv2.imwrite("thresh_output.jpg", thresh)
    cv2.imwrite("warped_output.jpg", warped)

    # Split into 81 cells
    cells = split_cells(warped)

    # Save sample cells (top-left 3x3 grid)
    for i in range(3):
        for j in range(3):
            cv2.imwrite(f"cell_{i}_{j}.jpg", cells[i][j])

    for i in range(3):
        for j in range(3):
            cleaned = clean_cell(cells[i][j])
            cv2.imwrite(f"clean_cell_{i}_{j}.jpg", cleaned)
    
    for i in range(3):
        for j in range(3):
            cleaned = clean_cell(cells[i][j])

            if is_cell_empty(cleaned):
                print(f"Cell {i},{j} is EMPTY")
            else:
                print(f"Cell {i},{j} has a digit")

    print("Saved:")
    print("- thresh_output.jpg")
    print("- contour_output.jpg")
    print("- warped_output.jpg")
    print("- sample cell images (cell_0_0.jpg ...)")
    print("Saved cleaned cells")


if __name__ == "__main__":
    main()