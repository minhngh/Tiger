"""Render questions as single-line or wrapped PNG images."""

import argparse
import os

from PIL import Image, ImageDraw, ImageFont

from tiger.common import image_path, index_records, read_records


SYSTEM_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def load_font(font_path=None, font_size=25):
    if font_path:
        return ImageFont.truetype(font_path, font_size)
    for candidate in SYSTEM_FONT_PATHS:
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, font_size)
            except OSError:
                continue
    return ImageFont.load_default()


def render_question(text, font, padding=20, width=None):
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    if width is None:
        # Preserve the original converter's sizing, offsets, and newline handling.
        bbox = measure.textbbox((0, 0), text, font=font)
        image = Image.new(
            "RGB",
            (bbox[2] - bbox[0] + 2 * padding, bbox[3] - bbox[1] + 2 * padding),
            "white",
        )
        ImageDraw.Draw(image).text((padding, padding), text, fill="black", font=font)
        return image

    lines = []
    for paragraph in text.splitlines() or [""]:
        line = ""
        # Character wrapping also handles long unbroken LaTeX expressions.
        for char in paragraph:
            if line and measure.textlength(line + char, font=font) > width:
                lines.append(line)
                line = ""
            line += char
        lines.append(line)
    rendered = "\n".join(lines)
    bbox = measure.multiline_textbbox((0, 0), rendered, font=font, spacing=6)
    image = Image.new(
        "RGB",
        (max(1, bbox[2] - bbox[0]) + 2 * padding,
         max(1, bbox[3] - bbox[1]) + 2 * padding),
        "white",
    )
    ImageDraw.Draw(image).multiline_text(
        (padding - bbox[0], padding - bbox[1]),
        rendered,
        font=font,
        fill="black",
        spacing=6,
    )
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--image-root", default="data")
    parser.add_argument("--layout", choices=["single-line", "wrapped"], default="single-line")
    parser.add_argument("--width", type=int, default=1200, help="Wrapped text width in pixels")
    parser.add_argument("--font", help="Font override; otherwise use the original system-font search")
    parser.add_argument("--font-size", type=int, default=25)
    parser.add_argument("--padding", type=int, default=20)
    args = parser.parse_args()
    if args.width < 1 or args.font_size < 1 or args.padding < 0:
        parser.error("width/font-size must be positive and padding nonnegative")
    font = load_font(args.font, args.font_size)
    rows = read_records(args.input)
    index_records(rows)
    paths = [image_path(args.image_root, row["image"]) for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("Multiple samples reference the same image path")
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    for row, path in zip(rows, paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        image = render_question(
            row["question"], font, args.padding,
            args.width if args.layout == "wrapped" else None,
        )
        with path.open("xb") as handle:
            image.save(handle, format="PNG")
    print(f"Rendered {len(rows)} images under {args.image_root}")


if __name__ == "__main__":
    main()
