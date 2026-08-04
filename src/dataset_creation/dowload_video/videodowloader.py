import sys
import yt_dlp
from yt_dlp.utils import download_range_func

def parse_time(time_str):
    """Converts HH:MM:SS or MM:SS string to seconds."""
    parts = [int(p) for p in time_str.split(':')]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]

def download_clip():
    url = input("Enter YouTube URL: ")
    start_time_str = input("Enter start time (HH:MM:SS): ")
    durata_str = input("Durata (seconds): ")
    
    filename_input = input("Enter output filename (e.g., clip): ").strip()
    # Ensure standard extension
    if not filename_input.endswith(".mp4"):
        filename_input += ".mp4"
    output = f"Dataset/video{filename_input}"

    start_sec = parse_time(start_time_str)
    end_sec = start_sec + int(durata_str)

    try:
        ydl_opts = {
            # Select 4K video + best audio
            'format': 'bestvideo[height<=2160]+bestaudio/best',
            
            # Download only the requested slice
            'download_ranges': download_range_func(None, [(start_sec, end_sec)]),
            'force_keyframes_at_cuts': True,
            
            # Force the final container to MP4
            'merge_output_format': 'mp4',
            
            # Apply re-encoding flags safely in post-processing
            'postprocessor_args': {
                'ffmpeg': [
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-c:a', 'aac'
                ]
            },
            
            'outtmpl': output,
            'cookiefile': 'src/youtube.com_cookies.txt',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        print(f"\nSuccessfully downloaded clip to {output}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    download_clip()