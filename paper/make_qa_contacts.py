"""Build four-page contact sheets from DOCX QA renders."""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def natural(path: Path) -> int:
    return int(path.stem.split("-")[-1])


def make(folder: Path) -> None:
    pages = sorted(folder.glob("page-*.png"), key=natural)
    contacts = folder / "contacts"
    contacts.mkdir(exist_ok=True)
    for start in range(0, len(pages), 4):
        group = pages[start:start + 4]
        sheet = Image.new("RGB", (1500, 2120), "#d8d8d8")
        draw = ImageDraw.Draw(sheet)
        for offset, page in enumerate(group):
            image = Image.open(page).convert("RGB")
            image.thumbnail((720, 990), Image.Resampling.LANCZOS)
            x = 20 + (offset % 2) * 745
            y = 50 + (offset // 2) * 1030
            sheet.paste(image, (x, y))
            draw.text((x, 15 + (offset // 2) * 1030), page.stem, fill="black")
        sheet.save(contacts / f"contact-{start + 1:02d}-{start + len(group):02d}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="*", default=["qa_attribution2_week4", "qa_attribution2_week10", "qa_attribution2_final"])
    args = parser.parse_args()
    for name in args.folders:
        make(ROOT / "tmp" / name)


if __name__ == "__main__":
    main()
