import cv2
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from reconnaisance2 import *
from torchvision import transforms



def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def center_digit(cell):
    _, thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cntr = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cntr)
        startx = (cell.shape[1] - w) // 2
        starty = (cell.shape[0] - h) // 2
        result = np.zeros_like(cell)
        result[starty:starty + h, startx:startx + w] = cell[y:y + h, x:x + w]
        return result, (w, h)
    return cell, (0, 0)

def process_sudoku_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Error loading image.")
        return None, None, None, None

    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_g, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            best_contour = contour

    if best_contour is None or len(best_contour) == 0:
        print("No suitable contour found.")
        return None, None, None, None

    best_contour = best_contour.reshape(-1, 2)
    warped = four_point_transform(img, best_contour)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_inverted = cv2.bitwise_not(warped_gray)

    sudoku_cells = []
    digit_sizes = []
    empty_cells = []
    rows = np.array_split(warped_inverted, 9, axis=0)
    for i, row in enumerate(rows):
        cols = np.array_split(row, 9, axis=1)
        for j, col in enumerate(cols):
            margin = int(0.2 * col.shape[0])
            cropped = col[margin:col.shape[0] - margin, margin:col.shape[1] - margin]
            cell = cv2.resize(cropped, (28, 28))
            kernel = np.ones((2, 2), np.uint8)
            cell = cv2.erode(cell, kernel, iterations=1)
            cell = cv2.dilate(cell, kernel, iterations=1)
            centered_cell, size = center_digit(cell)
            sudoku_cells.append(centered_cell)
            digit_sizes.append(size)

            if np.sum(centered_cell > 200) < 3:
                empty_cells.append((i, j))

    return warped, sudoku_cells, digit_sizes, empty_cells

def predict_digit(image, model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def get_sudoku_predictions(img_path, model):
    warped_image, sudoku_images, digit_sizes, empty_cells = process_sudoku_image(img_path)

    if sudoku_images is None:
        return None, None, None

    sudoku_predictions = []
    for idx, image in enumerate(sudoku_images):
        if image.shape != (28, 28):
            print("Error: Image is not 28x28 pixels")
            continue
        image = np.array(image)

        if np.sum(image > 200) < 3:
            sudoku_predictions.append(0)
        else:
            digit = predict_digit(image, model)
            sudoku_predictions.append(digit)

    sudoku_predictions = np.array(sudoku_predictions).reshape(9, 9)
    return sudoku_predictions, empty_cells, digit_sizes




def draw_predictions_on_image(original_image_path, empty_cells, digit_sizes, sudoku_matrix, output_image_path):
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        raise ValueError("L'image n'a pas pu être chargée. Vérifiez le chemin de l'image.")
    
    img_h, img_w = original_image.shape[:2]
    cell_h, cell_w = img_h // 9, img_w // 9

    for (i, j), (w, h) in zip(empty_cells, digit_sizes):
        digit = sudoku_matrix[i][j]
        if digit == 0:
            continue
        x = j * cell_w + cell_w // 2
        y = i * cell_h + cell_h // 2

        font_scale = 0.8  # Adjust the scale as needed 
        thickness = 2  # Adjust the thickness as needed

        text_size = cv2.getTextSize(str(digit), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2

        cv2.putText(original_image, str(digit), (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (22, 228, 50), thickness, cv2.LINE_AA)
    
    cv2.imwrite(output_image_path, original_image)
