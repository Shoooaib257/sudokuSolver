import cv2
import numpy as np


def preprocess_image(file):
    # Convert uploaded file to OpenCV image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold (important for varying lighting)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    return thresh

def find_largest_contour(thresh):
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No contours found")

    # Return the largest contour by area
    largest = max(contours, key=cv2.contourArea)

    return largest

def get_corners(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) != 4:
        raise ValueError("Could not find 4 corners")

    return approx

def order_points(pts):
    pts = pts.reshape(4, 2)

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

def warp_perspective(image, corners):
    rect = order_points(corners)

    (tl, tr, br, bl) = rect

    width = max(
        int(np.linalg.norm(br - bl)),
        int(np.linalg.norm(tr - tl))
    )

    height = max(
        int(np.linalg.norm(tr - br)),
        int(np.linalg.norm(tl - bl))
    )

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    return warped

def split_cells(warped):
    cells = []

    height, width = warped.shape[:2]
    cell_h = height // 9
    cell_w = width // 9

    for i in range(9):
        row = []
        for j in range(9):
            y1 = i * cell_h
            y2 = (i + 1) * cell_h
            x1 = j * cell_w
            x2 = (j + 1) * cell_w

            cell = warped[y1:y2, x1:x2]
            row.append(cell)

        cells.append(row)

    return cells

def clean_cell(cell):
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # Threshold
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    h, w = thresh.shape

    # Remove borders (very important)
    margin = int(min(h, w) * 0.2)
    cropped = thresh[margin:h - margin, margin:w - margin]

    return cropped

def is_cell_empty(cell):
    # Count white pixels
    total_pixels = cell.shape[0] * cell.shape[1]
    white_pixels = cv2.countNonZero(cell)

    # If very few white pixels → empty
    return white_pixels < total_pixels * 0.03