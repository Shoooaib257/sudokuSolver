import cv2
import numpy as np
from .ocr import recognize_digits_parallel

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

    # Filter for contours that could be the Sudoku grid
    # A Sudoku grid should be large and roughly square
    potential_grids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000: # Ignore very small contours
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        
        # We want something with 4 corners (or close to it)
        if len(approx) >= 4 and len(approx) <= 6:
            # Check aspect ratio
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 0.8 <= aspect_ratio <= 1.2:
                potential_grids.append(cnt)

    if not potential_grids:
        # Fallback to the largest contour if no square-like contour found
        return max(contours, key=cv2.contourArea)

    # Return the largest among the potential grids
    largest = max(potential_grids, key=cv2.contourArea)

    return largest

def get_corners(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) != 4:
        # Try a slightly different approximation if 4 corners not found
        approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
        if len(approx) != 4:
            raise ValueError(f"Could not find 4 corners, found {len(approx)}")

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

    # Use a fixed size for the warped image to make cell splitting consistent
    side = max(width, height)
    dst = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (side, side))

    return warped

def split_cells(warped):
    cells = []
    side = warped.shape[0]
    cell_side = side // 9

    for i in range(9):
        row = []
        for j in range(9):
            y1 = i * cell_side
            y2 = (i + 1) * cell_side
            x1 = j * cell_side
            x2 = (j + 1) * cell_side

            cell = warped[y1:y2, x1:x2]
            row.append(cell)
        cells.append(row)

    return cells

def clean_cell(cell):
    if len(cell.shape) == 3:
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell

    # Binary image (White digit, Black background)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    
    # Remove thin lines and small noise using opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    h, w = thresh.shape
    # Remove borders (reduce grid lines) - margin is critical
    margin = int(min(h, w) * 0.20)
    cropped = thresh[margin:h - margin, margin:w - margin]

    # Find contours in the cropped cell
    contours, _ = cv2.findContours(cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    
    # Debug for a specific cell that we expect to have a digit
    # if h > 0: # just to have access to i, j if we passed them, but we don't.
    
    # Relaxed HEURISTICS:
    area = cv2.contourArea(largest)
    x, y, w_box, h_box = cv2.boundingRect(largest)
    
    if area < (h * w * 0.03): # Relaxed from 0.05
        return None

    if h_box < (h * 0.25): # Relaxed from 0.3
        return None
    
    aspect_ratio = float(w_box) / h_box
    if aspect_ratio > 1.2 or aspect_ratio < 0.05: # Relaxed
        return None

    # Create a mask for the largest contour (the digit)
    mask = np.zeros_like(cropped)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    
    # Apply mask to cropped image
    digit_only = cv2.bitwise_and(cropped, mask)

    # Crop to the bounding box of the digit
    digit_img = digit_only[y:y + h_box, x:x + w_box]

    # Center the digit in a 28x28 square (MNIST style)
    # 1. Resize while maintaining aspect ratio, max 20px
    scale = 20.0 / max(w_box, h_box)
    digit_img = cv2.resize(digit_img, (int(w_box * scale), int(h_box * scale)))
    
    # 2. Place in center of 28x28
    h_new, w_new = digit_img.shape
    canvas = np.zeros((28, 28), dtype="uint8")
    y_off = (28 - h_new) // 2
    x_off = (28 - w_new) // 2
    canvas[y_off:y_off + h_new, x_off:x_off + w_new] = digit_img

    return canvas

def is_cell_empty(cell):
    # This is now handled within clean_cell returning None
    return cell is None

def extract_board(cells):
    non_empty_cells = []

    for i in range(9):
        for j in range(9):
            cleaned = clean_cell(cells[i][j])
            if cleaned is not None:
                print(f"Cell {i},{j} detected as NON-EMPTY")
                non_empty_cells.append((i, j, cleaned))
            else:
                # print(f"Cell {i},{j} detected as EMPTY")
                pass

    print(f"Total non-empty cells found: {len(non_empty_cells)}")
    # Use parallel OCR for non-empty cells
    board = recognize_digits_parallel(non_empty_cells)

    return board