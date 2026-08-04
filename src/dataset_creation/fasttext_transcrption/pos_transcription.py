import csv
import json
from pathlib import Path

import spacy


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = Path("data/transcription/transcriptions.csv")

TRANSCRIPTION_COLUMN = "transcription"
VERBS_COLUMN = "verbs"

SPACY_MODEL = "it_core_news_lg"


# ============================================================
# LOAD THE ITALIAN SPACY MODEL
# ============================================================

# The parser and NER components are not required for POS tagging
nlp = spacy.load(
    SPACY_MODEL,
    exclude=["parser", "ner"],
)


def extract_verbs(text: str) -> list[str]:
    """Extract unique verb lemmas from the provided text."""

    if not text or not text.strip():
        return []

    document = nlp(text)

    verbs = [
        token.lemma_.strip().lower()
        for token in document
        if token.pos_ in {"VERB", "AUX"}
        and token.is_alpha
        and token.lemma_.strip()
    ]

    # Remove duplicates while preserving the original order
    return list(dict.fromkeys(verbs))


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found: {CSV_PATH}"
        )

    # Read the original CSV
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

    # Add the verbs column if it does not already exist
    if VERBS_COLUMN not in fieldnames:
        fieldnames.append(VERBS_COLUMN)

    # Process every transcription
    for index, row in enumerate(rows, start=1):
        video_name = row.get("video_name", f"row {index}")
        transcription = row.get(TRANSCRIPTION_COLUMN, "")

        print(
            f"[{index}/{len(rows)}] "
            f"Processing {video_name}"
        )

        try:
            verbs = extract_verbs(transcription)

            # Store the list as JSON inside a single CSV cell
            row[VERBS_COLUMN] = json.dumps(
                verbs,
                ensure_ascii=False,
            )

        except Exception as error:
            print(f"Processing error: {error}")

            row[VERBS_COLUMN] = json.dumps(
                [],
                ensure_ascii=False,
            )

    # Write the updated data to a temporary file
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