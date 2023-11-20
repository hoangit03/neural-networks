import pygame
import numpy as np

# Khởi tạo Pygame
pygame.init()

# Kích thước mỗi ô trên màn hình
cell_size = 20

# Kích thước ma trận
rows, cols = 28, 28

# Tạo một mảng 2 chiều để lưu trữ giá trị của mỗi ô
grid = np.zeros((rows, cols))

# Màu sắc
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Kích thước màn hình
width, height = cols * cell_size, rows * cell_size + 50  # Thêm 50 đơn vị chiều cao cho phần nút

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Mouse Grid Input")

# Tạo nút reset
reset_button = pygame.Rect(0, rows * cell_size, width // 2, height - rows * cell_size)

hello_font = pygame.font.SysFont(None, 30)
hello_text = hello_font.render('Hello', True, (0, 0, 0))
hello_text_rect = hello_text.get_rect(center=(width//2, height-70)) 

font = pygame.font.Font(None, 36)

# Tạo nút classify
classify_button = pygame.Rect(width // 2, rows * cell_size, width // 2, height - rows * cell_size)

running = True
drawing = False  # Trạng thái nhấn giữ chuột

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Kiểm tra nút chuột trái được nhấn
                x, y = event.pos
                if reset_button.collidepoint(x, y):
                    
                    grid = np.zeros((rows, cols))
                elif classify_button.collidepoint(x, y):
                    print(np.array(grid)) 
                else:
                    drawing = True  # Bắt đầu nhấn giữ chuột để vẽ

        if event.type == pygame.MOUSEBUTTONUP:
            drawing = False  

        if event.type == pygame.MOUSEMOTION and drawing:
            x, y = event.pos
            row, col = y // cell_size, x // cell_size
            grid[row][col] = 1  

    # Vẽ nút reset
    pygame.draw.rect(screen, (200, 200, 200), reset_button)
    reset_font = pygame.font.SysFont(None, 30)
    reset_text = reset_font.render('Reset', True, (0, 0, 0))
    screen.blit(reset_text, (10, rows * cell_size + 10))
    screen.blit(hello_text, hello_text_rect)
    # Vẽ nút classify
    pygame.draw.rect(screen, (200, 200, 200), classify_button)
    classify_font = pygame.font.SysFont(None, 30)
    classify_text = classify_font.render('Classify', True, (0, 0, 0))
    screen.blit(classify_text, (width // 2 + 10, rows * cell_size + 10))

    # Vẽ lại màn hình dựa trên giá trị của mảng
    for row in range(rows):
        for col in range(cols):
            color = BLACK if grid[row][col] == 0 else RED
            pygame.draw.rect(screen, color, (col * cell_size, row * cell_size, cell_size, cell_size))

    # Cập nhật màn hình
    pygame.display.flip()

# Kết thúc game
pygame.quit()
