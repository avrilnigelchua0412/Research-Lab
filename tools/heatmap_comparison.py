import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
if __name__ == '__main__':
    paths = ['/workspace/Special_Problem/yolov5nano_heatmap_result/Pred', '/workspace/Special_Problem/yolov7tiny_heatmap_result/Pred']
    heads = ['GT', 'P3', 'P4', 'P5']
    for path in paths:
        version = path.split('/')[-2]
        for root, dirs, files in os.walk(path):
            for file in files:
                format = os.path.splitext(file)[1]
                tile = os.path.splitext(file)[0]
                level = root.split('/')[-1]
                if format in ['.jpeg', '.jpg', '.png']:
                    paths = []
                    path = os.path.join(root, file)
                    paths.append(path)
                    
                    for head in heads:
                        path = path.replace(path.split('/')[-3], head)
                        paths.append(path)
                        
                    images = [Image.open(p).convert("RGB") for p in paths]
                    total_width = sum(img.width for img in images)
                    canvas = Image.new("RGB", (total_width, 512), color=(0, 0, 0))
                    draw = ImageDraw.Draw(canvas)
                    x_offset = 0
                    padding = 10
                    for img in images:
                        canvas.paste(img, (x_offset, 0))
                        x_offset += img.width
                    dir = f"Special_Problem/combined_{version}/{level}"
                    file = f"{tile}{format}"
                    os.makedirs(dir, exist_ok=True)
                    save_path = os.path.join(dir, file)
                    canvas.save(save_path)