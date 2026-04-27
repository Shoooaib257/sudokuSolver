# Sudoku Solver (Image-Based)

A web application that takes an image of a Sudoku puzzle and returns the solved grid.

## Features
- Solve Sudoku from grid input
- (Upcoming) Solve Sudoku from image
- Clean backend architecture (Flask)

## Tech Stack
- Backend: Flask (Python)
- Image Processing: OpenCV (planned)
- OCR: Tesseract / CNN (planned)
- Frontend: HTML + JS (planned)

## API

### Solve Sudoku
POST /api/sudoku/solve

Request:
```json
{
  "board": [[...]]
}