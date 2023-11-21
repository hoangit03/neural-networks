import pygame
import os
from PIL import Image, ImageDraw

WIDTH = 400
HEIGHT = 400
DRAW_COLOR = (255, 255, 255)  # Màu nét vẽ
BG_COLOR = (0, 0, 0)  # Màu nền

# Khởi tạo Pygame
pygame.init()

# Tạo cửa sổ
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Vẽ và Lưu Nét Vẽ")

drawing = False
last_pos = None
radius = 13

# Tạo ảnh trống để vẽ
img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
draw = ImageDraw.Draw(img)

# Vẽ nút reset
font = pygame.font.Font(None, 36)
reset_text = font.render('Reset', True, (255, 255, 255))
reset_rect = reset_text.get_rect(center=(WIDTH // 2, HEIGHT - 50))

image_count = 2001

# Vòng lặp chính
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if reset_rect.collidepoint(event.pos):
                # Xóa hình ảnh và màn hình
                img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
                draw = ImageDraw.Draw(img)
                screen.fill(BG_COLOR)
            else:
                drawing = True
                pygame.draw.circle(screen, DRAW_COLOR, event.pos, radius)
                if last_pos:
                    draw.line([last_pos, event.pos], DRAW_COLOR, width=radius*2)
                last_pos = event.pos
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
            img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
            draw = ImageDraw.Draw(img)
            screen.fill(BG_COLOR)
        elif event.type == pygame.MOUSEMOTION and drawing:
            pygame.draw.circle(screen, DRAW_COLOR, event.pos, radius)
            if last_pos:
                draw.line([last_pos, event.pos], DRAW_COLOR, width=radius*2)
            last_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            last_pos = None
            
        
        if event.type == pygame.KEYDOWN and event.key == pygame.K_s and not drawing:
            # Lưu nét vẽ vào hình ảnh 28x28
            img_28x28 = img.resize((28, 28))
            folder_path = "image/pi"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_name = f"pi_{image_count}.png"  
            # file_name = f"aplpha_192.png" 
            file_path = os.path.join(folder_path, file_name)  
            img_28x28.save(file_path)
            image_count += 1

    # Vẽ nút reset lên màn hình


    pygame.display.flip()

# Kết thúc Pygame
pygame.quit()
