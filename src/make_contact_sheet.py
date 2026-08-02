import os
from PIL import Image
import math

img_dir = "data/synthetic_vehicle_icons_v2/images"
files = sorted(os.listdir(img_dir))

THUMB = 150
COLS = 12
rows = math.ceil(len(files) / COLS)

sheet = Image.new("RGB", (COLS * THUMB, rows * THUMB), (30, 30, 30))

for i, fname in enumerate(files):
    img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
    img.thumbnail((THUMB - 4, THUMB - 4))
    x = (i % COLS) * THUMB
    y = (i // COLS) * THUMB
    sheet.paste(img, (x + 2, y + 2))

sheet.save("data/synthetic_contact_sheet.jpg", quality=85)
print(f"Saved contact sheet with {len(files)} thumbnails to data/synthetic_contact_sheet.jpg")
print("Grid is 12 columns wide -- filename index N is at row N//12, column N%12")