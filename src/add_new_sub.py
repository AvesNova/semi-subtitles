import os
import subprocess

def add_subtitle_to_mkv(mkv_file, subtitle_file, output_file, language_code='eng', track_name='Konglish'):
    """
    Adds a subtitle file to an MKV file with the specified language code and track name.

    Args:
        mkv_file (str): Path to the MKV file.
        subtitle_file (str): Path to the subtitle file (SRT).
        language_code (str): Language code for the subtitle (default is 'eng').
        track_name (str): Name for the subtitle track (default is 'Konglish').
        output_file (str): Path to the output MKV file.

    Returns:
        None
    """
    cmd = [
        'mkvmerge', '-o', output_file,
        '--language', f'0:{language_code}',
        '--track-name', f'0:{track_name}',
        mkv_file, subtitle_file
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    mkv_file = 'Crash Landing on You\\Crash Landing on You (2019) - S01E01 - Episode 1 (1080p NF WEB-DL x265 MONOLITH).mkv'
    subtitle_file = 'output\\semi_translated_subtitles.srt'
    output_file = 'Crash Landing on You\\Crash Landing on You (2019) - S01E01 - Episode 1 (1080p NF WEB-DL x265 MONOLITH) - with Konglish subs.mkv'

    add_subtitle_to_mkv(mkv_file, subtitle_file, output_file)
    print(f"Subtitle added to MKV file: {output_file}")