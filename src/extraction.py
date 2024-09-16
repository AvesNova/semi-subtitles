import os
import subprocess
import re

def extract_subtitles(mkv_file, output_dir, languages=['eng', 'kor']):
    """
    Extracts specified language subtitles from an MKV file and saves them in the specified output directory.

    Args:
        mkv_file (str): Path to the MKV file.
        output_dir (str): Directory to save the extracted subtitle files.
        languages (list): List of language codes to extract (default is ['eng', 'kor']).

    Returns:
        list: List of paths to the extracted subtitle files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the list of subtitle tracks
    cmd = ['ffmpeg', '-i', mkv_file]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    output = result.stderr

    # Extract subtitle track information
    subtitle_tracks = []
    for line in output.split('\n'):
        if 'Subtitle' in line and any(lang in line for lang in languages):
            subtitle_tracks.append(line)

    # Extract subtitles
    extracted_files = []
    for i, track in enumerate(subtitle_tracks):
        # Extract track ID and language using regex
        track_id_match = re.search(r'Stream #0:(\d+)', track)
        lang_match = re.search(r'\((\w+)\)', track)
        if track_id_match and lang_match:
            track_id = track_id_match.group(1)
            lang = lang_match.group(1)
            output_file = os.path.join(output_dir, f'subtitle_{lang}_{i + 1}.srt')
            cmd = [
                'ffmpeg', '-i', mkv_file, '-map', f'0:{track_id}', '-c:s', 'srt',
                output_file
            ]
            subprocess.run(cmd)
            extracted_files.append(output_file)

    return extracted_files


if __name__ == '__main__':
    mkv_file = 'Crash Landing on You\\Crash Landing on You (2019) - S01E01 - Episode 1 (1080p NF WEB-DL x265 MONOLITH).mkv'  # Use double backslashes or raw strings for Windows paths
    output_dir = 'output'
    extracted_files = extract_subtitles(mkv_file, output_dir)
    print(f"Extracted subtitle files: {extracted_files}")