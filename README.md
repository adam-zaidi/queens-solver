# Queens Solver

This idea came to me in the shower at 4am, as all good ideas do. In my daily effort to lower my time in LinkedIn’s [Queens](https://www.linkedin.com/games/queens) game, I thought about how I could automate the process, and it lead here. This program uses opencv to automatically detect a Queens board via its colors, and pulls the gridlines & colors to get the board orientation. Then it uses a backtracking algorithm to find the correct placement for the queens, and double clicks them into place.

### Features

- **Screen Capture**: Use `mss` to take real-time screenshots of the game area.
- **Color Detection**: Utilize OpenCV to detect and differentiate colored regions on the puzzle grid.
- **Grid Analysis**: Models the puzzle grid using NumPy, and use a backtracking algorithm to find the correct placement.
- **Automated Interaction**: Uses `pynput` place the queens on the appropriate squares.
- **Visualization**: Display and save the processed grid with detected regions and placements using `matplotlib`. 

### ⚙️ Installation

Ensure you have Python installed. Then, install the required packages:

```bash
pip install opencv-python numpy mss pynput matplotlib
```

### A few notes if you want to run the program yourself:
1. The colors might appear slightly differently on your machine, so use pick_color.py to see how opencv samples the HSV colors of your display, and then change the bounds on LOWER_BLUE, UPPER_BLUE, LOWER_GREEN, and UPPER_GREEN.
2. Depending on the resolution of your screen, you will have to change SCALING_FACTOR so the coordinates are in the right place.
3. Then simply run the program and it should work fine!
