import numpy as np
import matplotlib.pyplot as plt


def drawLine(x0, y0, x1, y1):
    grid = np.zeros((100, 100))

    dx = x1 - x0
    dy = y1 - y0
    step = max(abs(dx), abs(dy))

    if step != 0:
        stepX = dx / step
        stepY = dy / step

        for i in range(step + 1):
            x = round(x0 + i * stepX)
            y = round(y0 + i * stepY)

            grid[x, y] = 1

    plt.imshow(grid.transpose(), cmap='gray')
    plt.show()


def bresenhams(x0, y0, x1, y1):
    grid = np.zeros((100, 100))

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = y1 - y0

    dir = 1 if dy > 0 else -1
    dy *= dir

    if dx != 0:
        y = y0
        p = 2 * dy - dx

        for i in range(dx + 1):
            grid[x0 + i, y] = 1

            if p >= 0:
                y += dir
                p -= 2 * dx

            p += 2 * dy

    plt.imshow(grid.transpose(), cmap='gray')
    plt.show()




def main():
    #drawLine(20, 10, 40, 99)
    #bresenhams(10, 30, 90, 70)
    #bresenhams(90, 70, 10, 30)
    bresenhams(50, 20, 70, 25)
    #bresenhams(10, 70, 90, 30)


if __name__ == '__main__':
    main()

