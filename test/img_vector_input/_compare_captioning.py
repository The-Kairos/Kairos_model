import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    ("BLIP", BASE_DIR / "blip_to_blip.py"),
    ("BLIP-2", BASE_DIR / "blip2_to_blip2.py"),
    ("CLIPCap", BASE_DIR / "clip_to_clipcap.py"),
]


def run_script(label: str, path: Path, image_path: Path | None) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing script: {path}")

    cmd = [sys.executable, str(path)]
    if image_path is not None:
        cmd.append(str(image_path))

    result = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed with code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{label} did not return JSON. STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        ) from exc

    return data


def format_seconds(value: float) -> str:
    return f"{value:.4f}"


def write_table(rows: list[dict], output_path: Path) -> None:
    lines = [
        "| Decoder | Caption | Encoder time (s) | Decoder time (s) |",
        "| --- | --- | --- | --- |",
    ]

    for row in rows:
        caption = row["caption"].replace("|", "\\|")
        lines.append(
            "| "
            + " | ".join(
                [
                    row["decoder"],
                    caption,
                    format_seconds(row["embedding_time"]),
                    format_seconds(row["caption_time"]),
                ]
            )
            + " |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    image_path = None
    if len(sys.argv) > 1:
        raw_path = Path(sys.argv[1]).expanduser()
        if raw_path.is_absolute():
            image_path = raw_path.resolve()
        else:
            image_path = (Path.cwd() / raw_path).resolve()
    rows = []
    for label, path in SCRIPTS:
        data = run_script(label, path, image_path)
        rows.append(
            {
                "decoder": label,
                "caption": data.get("caption", ""),
                "embedding_time": float(data.get("embedding_time", 0.0)),
                "caption_time": float(data.get("caption_time", 0.0)),
            }
        )

    output_path = BASE_DIR / "_table_comparison.md"
    write_table(rows, output_path)


if __name__ == "__main__":
    main()
