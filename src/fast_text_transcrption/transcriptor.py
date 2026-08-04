import os
import csv
from pathlib import Path

# Usa la GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from faster_whisper import WhisperModel


# ============================================================
# CONFIGURAZIONE
# ============================================================

# Cartella contenente tutti i video
CARTELLA_VIDEO = Path(r"MAIA-AV/video")

# CSV che verrà creato
FILE_CSV = Path(r"MAIA-AV/output/transcription/trascrizioni.csv")

# Formati video da elaborare
ESTENSIONI_VIDEO = {
    ".mp4",
}

# Modello Whisper
NOME_MODELLO = "large-v3"

# "it" per forzare l'italiano; None per rilevamento automatico
LINGUA = "it"


# ============================================================
# CARICAMENTO MODELLO SULLA GPU 0
# ============================================================

model = WhisperModel(
    NOME_MODELLO,
    device="cuda",
    device_index=0,
    compute_type="float16",
)


def trova_video(cartella: Path) -> list[Path]:
    """
    Restituisce tutti i video presenti nella cartella.

    Usa rglob per cercare anche nelle sottocartelle.
    """
    return sorted(
        percorso
        for percorso in cartella.rglob("*")
        if percorso.is_file()
        and percorso.suffix.lower() in ESTENSIONI_VIDEO
    )


def trascrivi_video(percorso_video: Path) -> str:
    """
    Trascrive l'audio di un singolo video.
    """
    segments, info = model.transcribe(
        str(percorso_video),
        language=LINGUA,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    parti_trascrizione = []

    for segment in segments:
        testo = segment.text.strip()

        if testo:
            parti_trascrizione.append(testo)

    return " ".join(parti_trascrizione)


def main() -> None:
    if not CARTELLA_VIDEO.exists():
        raise FileNotFoundError(
            f"La cartella non esiste: {CARTELLA_VIDEO}"
        )

    video_trovati = trova_video(CARTELLA_VIDEO)

    if not video_trovati:
        print(f"Nessun video trovato in: {CARTELLA_VIDEO}")
        return

    print(f"Video trovati: {len(video_trovati)}")

    FILE_CSV.parent.mkdir(parents=True, exist_ok=True)

    with FILE_CSV.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "nome_video",
                "percorso_video",
                "trascrizione",
            ],
            delimiter=";",
        )

        writer.writeheader()

        for indice, percorso_video in enumerate(
            video_trovati,
            start=1,
        ):
            print(
                f"[{indice}/{len(video_trovati)}] "
                f"Trascrizione di {percorso_video.name}"
            )

            try:
                trascrizione = trascrivi_video(percorso_video)

                writer.writerow(
                    {
                        "nome_video": percorso_video.name,
                        "percorso_video": str(percorso_video),
                        "trascrizione": trascrizione,
                    }
                )

                csv_file.flush()

                if trascrizione:
                    print("Trascrizione completata.")
                else:
                    print("Nessun parlato rilevato.")

            except Exception as errore:
                print(f"Errore durante la trascrizione: {errore}")

                writer.writerow(
                    {
                        "nome_video": percorso_video.name,
                        "percorso_video": str(percorso_video),
                        "trascrizione": f"ERRORE: {errore}",
                    }
                )

                csv_file.flush()

    print(f"\nCSV creato: {FILE_CSV}")


if __name__ == "__main__":
    main()