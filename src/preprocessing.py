import re
import pysrt
import json
from openai import OpenAI
import tiktoken
from pysrt import SubRipTime, SubRipFile, SubRipItem
from dataclasses import dataclass
import concurrent.futures

@dataclass
class MergedSubtitle:
    index: int
    start: SubRipTime
    end: SubRipTime
    korean: str
    english: str
    type: str

def read_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read().strip()

def load_subtitles(file_path: str) -> SubRipFile:
    return pysrt.open(file_path)

def clean_subtitle_text(text: str) -> str:
    return re.sub(r'(\{.*?\}|\[.*?\]|<.*?>|\(.*?\))', '', text).strip()

def is_closed_caption(text: str) -> bool:
    return re.match(r'^\s*(\{.*?\}|\[.*?\]|<.*?>|\(.*?\))\s*$', text) is not None

def overlaps(sub1: SubRipItem, sub2: SubRipItem) -> bool:
    return sub1.start <= sub2.end and sub2.start <= sub1.end

def merge_overlapping(korean: SubRipItem, english_subs: list[SubRipItem]) -> MergedSubtitle:
    overlapping = [e for e in english_subs if overlaps(korean, e)]
    end_time = max([korean.end] + [e.end for e in overlapping])
    return MergedSubtitle(
        index=korean.index,
        start=korean.start,
        end=end_time,
        korean=korean.text,
        english=' '.join(e.text for e in overlapping),
        type='combined' if overlapping else 'korean'
    )

def merge_subtitles(korean_subs: list[SubRipItem], english_subs: list[SubRipItem]) -> list[MergedSubtitle]:
    merged = []
    k_index, e_index = 0, 0

    while k_index < len(korean_subs) or e_index < len(english_subs):
        if k_index < len(korean_subs) and (e_index == len(english_subs) or korean_subs[k_index].start <= english_subs[e_index].start):
            k_sub = korean_subs[k_index]
            merged.append(merge_overlapping(k_sub, english_subs[e_index:]))
            k_index += 1
            while e_index < len(english_subs) and english_subs[e_index].start <= k_sub.end:
                e_index += 1
        else:
            e_sub = english_subs[e_index]
            if k_index < len(korean_subs) and overlaps(e_sub, korean_subs[k_index]):
                e_index += 1
                continue
            merged.append(MergedSubtitle(
                index=e_sub.index,
                start=e_sub.start,
                end=e_sub.end,
                korean='',
                english=e_sub.text,
                type='english'
            ))
            e_index += 1

    merged.sort(key=lambda x: x.start)
    for i, sub in enumerate(merged, start=1):
        sub.index = i

    return merged

def preprocess_subtitles(merged_subs: list[MergedSubtitle]) -> list[dict]:
    processed = []
    
    for sub in merged_subs:
        if sub.type == 'english':
            processed.append({
                'index': sub.index,
                'start': str(sub.start),
                'end': str(sub.end),
                'full english translation': sub.english,
                'type': 'english'
            })
        else:
            if is_closed_caption(sub.korean):
                continue
            
            cleaned_korean = clean_subtitle_text(sub.korean)
            
            if not cleaned_korean:
                continue
            
            duration = (sub.end - sub.start).seconds
            processed.append({
                'index': sub.index,
                'start': str(sub.start),
                'end': str(sub.end),
                'duration': f"{duration:.1f} seconds",
                'original korean': cleaned_korean,
                'full english translation': sub.english,
                'type': sub.type
            })
    
    return processed

def num_tokens_from_string(string: str, model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_llm_response(chunk: dict, client: OpenAI, messages: list[dict], model_name: str = "gpt-4o-mini") -> tuple[dict | None, list[dict]]:
    input_json = json.dumps(chunk, ensure_ascii=False)
    input_message = f"\n\nInput:\n{input_json}\n\nOutput:"
    messages.append({"role": "user", "content": input_message})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    
    output = response.choices[0].message.content
    print(f"Processing chunk {chunk['index']}:")
    print(output)
    print()

    try:
        json_output = re.search(r'{[\s\S]*}', output).group(0)
        parsed_output = json.loads(json_output)
        
        for key, value in parsed_output.items():
            if isinstance(value, str):
                parsed_output[key] = value.strip('"')
        
        parsed_output['start'] = chunk['start']
        parsed_output['end'] = chunk['end']
        messages.append({"role": "assistant", "content": output})
        return parsed_output, messages
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing JSON for chunk {chunk['index']}: {e}")
        return None, messages

class SubtitleProcessor:
    def __init__(self, api_key: str, prompt_template: str, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.prompt_template = prompt_template
        self.model_name = model_name
        self.cache = {}  # Simple in-memory cache

    def process_chunk(self, chunk: dict) -> dict | None:
        if chunk['type'] == 'english':
            return {
                'start': chunk['start'],
                'end': chunk['end'],
                'semi translation': chunk['full english translation']
            }

        # Check cache first
        cache_key = json.dumps(chunk)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Process with LLM
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": self.prompt_template}
        ]
        processed_chunk, _ = get_llm_response(chunk, self.client, messages, self.model_name)
        
        # Cache the result
        if processed_chunk:
            # Remove the 'index' key from the processed chunk
            processed_chunk.pop('index', None)
            self.cache[cache_key] = processed_chunk
        return processed_chunk

    def process_chunks(self, chunks: list[dict]) -> list[dict | None]:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(self.process_chunk, chunks))

def reassemble_srt(processed_chunks: list[dict | None], output_srt_file: str) -> None:
    subs = SubRipFile()
    
    for index, chunk in enumerate(processed_chunks, start=1):
        if chunk:  # Check if chunk is not None
            sub = SubRipItem(
                index=index,  # Use the new index
                start=SubRipTime.from_string(chunk['start']),
                end=SubRipTime.from_string(chunk['end']),
                text=chunk['semi translation']
            )
            subs.append(sub)
    
    subs.save(output_srt_file, encoding='utf-8')

def main():
    korean_srt_file = 'raw_srt\\subtitle_kor_2.srt'
    english_srt_file = 'raw_srt\\subtitle_eng_1.srt'
    output_srt_file = 'output\\semi_translated_subtitles.srt'
    api_key_file = 'keys\\openai.txt'
    prompt_file = 'src\\prompt.txt'
    
    api_key = read_file(api_key_file)
    prompt_template = read_file(prompt_file)
    
    korean_subs = load_subtitles(korean_srt_file)
    english_subs = load_subtitles(english_srt_file)
    
    merged_subs = merge_subtitles(korean_subs, english_subs)
    preprocessed_chunks = preprocess_subtitles(merged_subs)
    
    processor = SubtitleProcessor(api_key, prompt_template)
    processed_chunks = processor.process_chunks(preprocessed_chunks)
    
    # Print all processed chunks
    for chunk in processed_chunks:
        if chunk:
            print(json.dumps(chunk, ensure_ascii=False, indent=2))
            print()
    
    reassemble_srt(processed_chunks, output_srt_file)
    print(f"Semi-translated subtitles saved to {output_srt_file}")

if __name__ == '__main__':
    main()