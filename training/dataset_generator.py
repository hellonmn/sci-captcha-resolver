"""Generate independent synthetic character datasets for arithmetic glyphs."""
from pathlib import Path
import argparse
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

CLASSES = list("0123456789+-×÷=")
SPLITS = {"train": 12000, "validation": 2000, "test": 2000}
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/comic.ttf", "C:/Windows/Fonts/segoepr.ttf",
    "C:/Windows/Fonts/Inkfree.ttf",
]

def fonts():
    return [Path(p) for p in FONT_CANDIDATES if Path(p).is_file()]

def generate_one(char, rng, available_fonts, size=32):
    bg = rng.choice([0, 8, 16, 240, 248, 255])
    fg = rng.choice([0, 12, 32, 224, 244, 255])
    if abs(bg - fg) < 100:
        fg = 255 - bg
    canvas = Image.new("L", (size, size), bg)
    draw = ImageDraw.Draw(canvas)
    fp = rng.choice(available_fonts) if available_fonts else None
    font_size = rng.randint(17, 29)
    font = ImageFont.truetype(str(fp), font_size) if fp else ImageFont.load_default()
    box = draw.textbbox((0, 0), char, font=font, stroke_width=rng.choice([0, 0, 1]))
    x = (size - (box[2] - box[0])) / 2 - box[0] + rng.randint(-4, 4)
    y = (size - (box[3] - box[1])) / 2 - box[1] + rng.randint(-4, 4)
    draw.text((int(x), int(y)), char, font=font, fill=fg)
    # A substantial handwritten-style '+' subset teaches the classifier that
    # tilted, uneven crosses are still addition signs.
    if char == "+" and rng.random() < 0.7:
        canvas.paste(bg, (0, 0, size, size))
        draw = ImageDraw.Draw(canvas)
        cx = size // 2 + rng.randint(-2, 2)
        cy = size // 2 + rng.randint(-2, 2)
        half = rng.randint(6, 11)
        stroke = rng.randint(2, 5)
        tilt = rng.randint(-3, 3)
        draw.line((cx - half, cy + rng.randint(-2, 2), cx + half, cy + rng.randint(-2, 2)), fill=fg, width=stroke)
        draw.line((cx + rng.randint(-2, 2), cy - half, cx + tilt, cy + half), fill=fg, width=stroke)
    elif char == "-" and rng.random() < 0.7:
        canvas.paste(bg, (0, 0, size, size))
        draw = ImageDraw.Draw(canvas)
        cy = size // 2 + rng.randint(-3, 3)
        half = rng.randint(6, 12)
        stroke = rng.randint(2, 5)
        slant = rng.randint(-3, 3)
        draw.line((size // 2 - half, cy + slant, size // 2 + half, cy - slant), fill=fg, width=stroke)
    canvas = canvas.rotate(rng.uniform(-15, 15), resample=Image.Resampling.BILINEAR, fillcolor=bg)
    if rng.random() < 0.35:
        # Pillow's native noise effect avoids a NumPy dependency in data generation.
        noise = Image.effect_noise(canvas.size, rng.uniform(5, 24))
        canvas = Image.blend(canvas, noise, 0.08)
    if rng.random() < 0.15:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.6)))
    if rng.random() < 0.25:
        draw = ImageDraw.Draw(canvas)
        for _ in range(rng.randint(1, 4)):
            px, py = rng.randrange(size), rng.randrange(size)
            draw.point((px, py), fill=rng.choice([bg, fg]))
    return canvas

def generate(output, counts=SPLITS, seed=1729):
    output = Path(output)
    available_fonts = fonts()
    if not available_fonts:
        print("Warning: using Pillow's built-in font; install system fonts for more variety")
    for split_idx, (split, count) in enumerate(counts.items()):
        # Independent RNG stream per split prevents shared examples across splits.
        rng = random.Random(seed + split_idx * 1000003)
        for class_idx, char in enumerate(CLASSES):
            # Zero padding keeps ImageFolder's alphabetical order aligned with CLASSES.
            directory = output / split / f"{class_idx:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                image = generate_one(char, rng, available_fonts)
                image.save(directory / f"{i:06d}.png", optimize=True)
    (output / "classes.txt").write_text("\n".join(CLASSES), encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/generated")
    parser.add_argument("--train", type=int, default=SPLITS["train"])
    parser.add_argument("--validation", type=int, default=SPLITS["validation"])
    parser.add_argument("--test", type=int, default=SPLITS["test"])
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    generate(args.output, {"train": args.train, "validation": args.validation, "test": args.test}, args.seed)
