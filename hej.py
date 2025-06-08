import numpy as np
from PIL import Image
import os

rows = 30
cols = 22
tile_size = 224  # patch boyutu

height = rows * tile_size
width = cols * tile_size

big_image = np.zeros((height, width), dtype=np.uint8)  # Tek kanallı, 8 bitlik boş büyük resim

folder = "predictions"

# Dosya isimlerini sırayla alıyoruz
all_filenames = []
for idx in range(690):
    filename = os.path.join(folder, f"0_{idx}.tif")
    if os.path.exists(filename):
        all_filenames.append(filename)

if len(all_filenames) < rows * cols:
    print(f"UYARI: Yeterli dosya yok! Sadece {len(all_filenames)} dosya bulundu.")

file_idx = 0  # kullanılacak dosya indeksi

for row in range(rows):
    for col in range(cols):
        if file_idx >= len(all_filenames):
            print("Dosya kalmadı, döngü durduruluyor.")
            break

        filename = all_filenames[file_idx]
        tile = np.array(Image.open(filename))

        if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
            print(f"Uyarı: {filename} boyutu beklenen {tile_size}x{tile_size} değil, {tile.shape}")

        start_y = row * tile_size
        start_x = col * tile_size

        big_image[start_y:start_y + tile_size, start_x:start_x + tile_size] = tile

        file_idx += 1
    else:
        continue
    break

# Sonuç kaydet
Image.fromarray(big_image).save("merged_prediction.tif")
print("Birleştirme tamamlandı, merged_prediction.tif olarak kaydedildi.")
