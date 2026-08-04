import csv
import json
from pathlib import Path

import spacy


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = Path("data/transcription/transcriptions.csv")

TRANSCRIPTION_COLUMN = "transcription"
COMMON_NOUNS_COLUMN = "common_nouns"

MODEL_NAME = "it_core_news_lg"


# ============================================================
# LOAD THE ITALIAN NLP PIPELINE
# ============================================================

# The parser and NER components are not needed for noun extraction
nlp = spacy.load(
    MODEL_NAME,
    exclude=["parser", "ner"],
)


def extract_common_nouns(text: str) -> list[str]:
    """Extract unique common-noun lemmas from a text."""

    if not text or not text.strip():
        return []

    doc = nlp(text)

    nouns = [
        token.lemma_.strip().lower()
        for token in doc
        if token.pos_ == "NOUN"
        and token.is_alpha
        and token.lemma_.strip()
    ]

    # Remove duplicates while preserving their original order
    return list(dict.fromkeys(nouns))


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    with CSV_PATH.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file,
            delimiter=";",
        )

        if not reader.fieldnames:
            raise ValueError("The CSV file has no header.")

        if TRANSCRIPTION_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"Column '{TRANSCRIPTION_COLUMN}' not found. "
                f"Available columns: {reader.fieldnames}"
            )

        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    if COMMON_NOUNS_COLUMN not in fieldnames:
        fieldnames.append(COMMON_NOUNS_COLUMN)

    for index, row in enumerate(rows, start=1):
        video_name = row.get("video_name", f"row {index}")
        transcription = row.get(TRANSCRIPTION_COLUMN, "")

        print(f"[{index}/{len(rows)}] Processing {video_name}")

        try:
            common_nouns = extract_common_nouns(transcription)

            # Store the Python list as JSON inside one CSV cell
            row[COMMON_NOUNS_COLUMN] = json.dumps(
                common_nouns,
                ensure_ascii=False,
            )

        except Exception as error:
            print(f"Processing error: {error}")

            row[COMMON_NOUNS_COLUMN] = json.dumps(
                [],
                ensure_ascii=False,
            )

    temporary_csv = CSV_PATH.with_suffix(".temporary.csv")

    with temporary_csv.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            delimiter=";",
            quotechar='"',
            quoting=csv.QUOTE_ALL,
        )

        writer.writeheader()
        writer.writerows(rows)

    # Replace the original CSV only after successful completion
    temporary_csv.replace(CSV_PATH)

    print(f"\nCSV updated: {CSV_PATH}")


if __name__ == "__main__":
    main()