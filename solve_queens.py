import cv2
import numpy as np
import mss
import time
import os
from pynput.mouse import Button, Controller, Listener

# CONFIG
LOWER_BLUE = (105, 65, 235)
UPPER_BLUE = (115, 100, 255)
LOWER_GREEN = (45, 55, 210)
UPPER_GREEN = (50, 80, 255)
SCALING_FACTOR = 0.5 # depending on your display, might need to change this
DELAY = 0.1 # any less than this will cause a blank queens result, as it can't display a 0:00 time

mouse = Controller()
directory = f"outputs/output{int(time.time())}"
os.mkdir(directory)

# SOME MOUSE STUFF

def on_click(x, y, button, pressed):
    if pressed:
        mouse.position = (0,0)
        return False

listener = Listener(on_click=on_click)
listener.start()

# COLOR STUFF
def hex_to_bgr(hex_str):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) != 6:
        raise ValueError(f"Invalid hex color: '{hex_str}'")
    rgb = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return rgb[::-1]  # BGR

def colors_detected(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, np.array(LOWER_BLUE), np.array(UPPER_BLUE))
    mask_green = cv2.inRange(hsv, np.array(LOWER_GREEN), np.array(UPPER_GREEN))
    return cv2.countNonZero(mask_blue) > 50 and cv2.countNonZero(mask_green) > 50

# SCREENSHOT AND BOARD DETECTION
board_image = None
with mss.mss() as sct:
    monitor = sct.monitors[1]
    board_detected = False

    while True:
        img = np.array(sct.grab(monitor))
        board_image = img

        if colors_detected(img):
            filename = os.path.join(directory, f"screenshot.png")
            cv2.imwrite(filename, img)
            print("✅ Queens Board Detected")
            start = time.time()
            break
        elif not board_detected:
            print("⛔ No board detected.")
            board_detected = True

        if cv2.waitKey(1) == 27:
            break

# FIND DA GRID
def find_grid():
    img = board_image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(directory, "gray.png"), gray)

    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(os.path.join(directory, "binary.png"), binary)

    edges = cv2.Canny(binary, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_offset, y_offset, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    grid_only = img[y_offset:y_offset+h, x_offset:x_offset+w]
    cv2.imwrite(os.path.join(directory, "grid_only.png"), grid_only)

    gray_crop = cv2.cvtColor(grid_only, cv2.COLOR_BGR2GRAY)
    _, bin_crop = cv2.threshold(gray_crop, 50, 255, cv2.THRESH_BINARY_INV)
    edges_crop = cv2.Canny(bin_crop, 50, 150)

    line_image = np.copy(grid_only)
    lines = cv2.HoughLinesP(edges_crop, 1, np.pi / 180, 100, 30, 5)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    line_edges = cv2.addWeighted(grid_only, 0.8, line_image, 1, 0)
    cv2.imwrite(os.path.join(directory, "lines.png"), line_edges)

    x_lines, y_lines = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            (x_lines if abs(x1 - x2) < 10 else y_lines).append((x1 + x2) // 2 if abs(x1 - x2) < 10 else (y1 + y2) // 2)

    def cluster(vals, eps=10):
        vals.sort()
        clustered, group = [], [vals[0]]
        for v in vals[1:]:
            if abs(v - group[-1]) < eps:
                group.append(v)
            else:
                clustered.append(int(np.mean(group)))
                group = [v]
        clustered.append(int(np.mean(group)))
        return clustered

    x_lines = cluster(x_lines) if x_lines else []
    y_lines = cluster(y_lines) if y_lines else []
    if len(x_lines) < 2 or len(y_lines) < 2:
        raise ValueError("Not enough grid lines found.")

    num_cols, num_rows = len(x_lines) - 1, len(y_lines) - 1
    cell_width = int(np.mean(np.diff(x_lines)))
    cell_height = int(np.mean(np.diff(y_lines)))
    offset_x = int(cell_width / 5)
    offset_y = int(cell_height / 5)
    color_grid = np.empty((num_cols, num_rows), dtype='<U7')
    colors = {}

    for row in range(num_rows):
        for col in range(num_cols):
            cx, cy = x_lines[col] + offset_x, y_lines[row] + offset_y
            b, g, r, a = grid_only[cy, cx]
            hex_color = f'#{r:02X}{g:02X}{b:02X}'
            color_grid[row][col] = hex_color
            colors.setdefault(hex_color, []).append((col, row))

    return num_cols, num_rows, cell_width, cell_height, colors, color_grid, (x_offset, y_offset), (x_lines, y_lines)

# Find the queens
def find_neighbors(x, y, size):
    return [(i, y) for i in range(size) if i != x] + [(x, i) for i in range(size) if i != y] + [
        (x + dx, y + dy)
        for dx in [-1, 0, 1] for dy in [-1, 0, 1]
        if (dx != 0 or dy != 0) and 0 <= x + dx < size and 0 <= y + dy < size
    ]

def is_valid(x, y, states):
    return states[x, y] == 0 and all(states[nx, ny] == 0 for nx, ny in find_neighbors(x, y, len(states)))

def solve(colors_sorted, colors, states, color_index=0, result={}, final_state=None):
    if color_index == len(colors_sorted):
        final_state[0] = states.copy()
        return True

    color = colors_sorted[color_index]
    for x, y in colors[color]:
        if is_valid(x, y, states):
            states[x, y], result[color] = 1, (x, y)
            if solve(colors_sorted, colors, states, color_index + 1, result, final_state):
                return True
            states[x, y] = 0
            del result[color]
    return False

def solve_queen():
    num_cols, num_rows, cell_width, cell_height, colors, color_grid, offsets, lines = find_grid()
    states = np.zeros((num_cols, num_rows), dtype=int)
    sorted_colors = sorted(colors, key=lambda c: len(colors[c]))
    result, final_state = {}, [None]

    if not solve(sorted_colors, colors, states, 0, result, final_state):
        raise ValueError("No valid configuration found.")

    cell_size = 50
    img2 = np.zeros((cell_size * num_rows, cell_size * num_cols, 3), dtype=np.uint8)
    for i in range(num_rows):
        for j in range(num_cols):
            color = hex_to_bgr(color_grid[i, j])
            cv2.rectangle(img2, (j*cell_size, i*cell_size), ((j+1)*cell_size, (i+1)*cell_size), color, -1)
            if final_state[0][j][i]:
                cv2.circle(img2, (j*cell_size + cell_size//2, i*cell_size + cell_size//2), 10, (0, 0, 0), -1)

    cv2.imwrite(os.path.join(directory, 'queens_solution.png'), img2)

    # double click the queens
    marked = board_image.copy()
    for color, (row, col) in result.items():
        cx = lines[1][row] + int(cell_width / 2)
        cy = lines[0][col] + int(cell_height / 2)
        screen_x = int((offsets[0] + cx) * SCALING_FACTOR)
        screen_y = int((offsets[1] + cy) * SCALING_FACTOR)
        mouse.position = (screen_x, screen_y)
        if DELAY != 0: time.sleep(DELAY)
        mouse.click(Button.left, 2)
        print(f"Clicked cell ({row}, {col}) at ({screen_x}, {screen_y})")
        cv2.circle(marked, (screen_x, screen_y), 10, (0, 0, 255), -1)

    return final_state[0], result

# RUN
solve_queen()
end = time.time()

print(f"Total runtime of the program is {end - start} seconds")