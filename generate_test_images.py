#!/usr/bin/env python3
import json
import os
import random
from PIL import Image, ImageDraw, ImageFont

CHARACTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FONT_SIZE = 16
FONT_PATH = 'font/font/OCR-B.ttf'
OUTPUT_DIR = 'output_test'
NUM_IMAGES = 10  # テスト用に少量

def get_text_bbox(draw, text, font, x, y):
    bbox = draw.textbbox((x, y), text, font=font)
    return {
        'x': bbox[0],
        'y': bbox[1],
        'w': bbox[2] - bbox[0],
        'h': bbox[3] - bbox[1],
        'char': text  # 文字の内容を追加
    }

def generate_image(image_index):
    img = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    
    num_chars = random.randint(1, 10)  # テスト用に少なめ
    bounding_boxes = []
    
    occupied_regions = []
    
    for _ in range(num_chars):
        char = random.choice(CHARACTERS)
        
        temp_bbox = draw.textbbox((0, 0), char, font=font)
        char_width = temp_bbox[2] - temp_bbox[0]
        char_height = temp_bbox[3] - temp_bbox[1]
        
        max_attempts = 100
        for attempt in range(max_attempts):
            x = random.randint(0, IMAGE_WIDTH - char_width - 1)
            y = random.randint(0, IMAGE_HEIGHT - char_height - 1)
            
            new_region = (x, y, x + char_width, y + char_height)
            
            overlaps = False
            for region in occupied_regions:
                if not (new_region[2] < region[0] or 
                       new_region[0] > region[2] or 
                       new_region[3] < region[1] or 
                       new_region[1] > region[3]):
                    overlaps = True
                    break
            
            if not overlaps:
                draw.text((x, y), char, fill='black', font=font)
                bbox = get_text_bbox(draw, char, font, x, y)
                bounding_boxes.append(bbox)
                occupied_regions.append(new_region)
                break
    
    image_filename = f"{image_index:05d}.png"
    json_filename = f"bb_{image_index:05d}.json"
    
    img.save(os.path.join(OUTPUT_DIR, image_filename))
    
    with open(os.path.join(OUTPUT_DIR, json_filename), 'w') as f:
        json.dump(bounding_boxes, f, indent=2)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Generating {NUM_IMAGES} test images...")
    for i in range(1, NUM_IMAGES + 1):
        generate_image(i)
        print(f"Generated image {i}")
    
    # 最初のJSONファイルの内容を確認
    with open(os.path.join(OUTPUT_DIR, 'bb_00001.json'), 'r') as f:
        data = json.load(f)
        print("\n最初のJSONファイルの内容:")
        print(json.dumps(data[:3], indent=2))  # 最初の3つだけ表示

if __name__ == "__main__":
    main()