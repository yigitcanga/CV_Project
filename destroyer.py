# import os
#
# folder = "dataset/train/images_after"  # dosyaların olduğu klasör yolu
#
# for idx in range(690):  # 0'dan 689'a kadar
#     # İlgili dosya adı
#     filename = f"0_{idx}.tif"
#     filepath = os.path.join(folder, filename)
#
#     # Eğer dosya indeksi 22 mod 23 ise (her 23 dosyada bir), dosyayı sil
#     if idx % 23 == 22:
#         if os.path.exists(filepath):
#             os.remove(filepath)
#             print(f"Silindi: {filename}")
#         else:
#             print(f"Dosya bulunamadı, silinemedi: {filename}")

import os

folder = "dataset/test/images_before"
files = os.listdir(folder)

for f in files:
    # Sadece tif dosyaları için
    if f.endswith(".tif"):
        # Dosya ismini sayısal olarak al (örneğin '1.tif' -> 1)
        num = int(os.path.splitext(f)[0])
        new_name = f"{num:04d}.tif"  # 4 basamaklı, başına sıfır ekler
        os.rename(os.path.join(folder, f), os.path.join(folder, new_name))