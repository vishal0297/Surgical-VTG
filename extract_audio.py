import torch
import numpy as np
from transformers import WhisperForConditionalGeneration, AutoProcessor, pipeline
import os
import json
import pydub
import whisper
# import whisperx
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent, detect_silence
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


def extract_audio(inputData_folder, output_format = "mp3", output_folder=None):
    inputData = os.listdir(inputData_folder+"video/")
    if output_folder is None:
        output_folder = inputData_folder+"raw_audio/"
        os.makedirs(output_folder)
    for data in inputData:
        name, data_format = (data.split("."))
        print(data_format)
        audio = AudioSegment.from_file(inputData_folder+"video/"+data, data_format)
        audio.export(output_folder+name+"."+output_format,output_format)

def load_openai_whisper():
    model = whisper.load_model("large-v3", device=device)
    # options = whisper.DecodingOptions(language="en")
    return model

def load_whisperx_model():
    model = whisperx.load_model("large-v2", device, compute_type="float16")

    return model

def load_model():
    model_id = "openai/whisper-large-v3"

    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch_dtype)
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device
)
    return pipe

def extract_text_from_mp3(inputData_folder, output_folder=None):
    s2tpipe = load_model()
    # model = load_whisperx_model()
    # model = load_openai_whisper()
    # transcribe_options = dict(language="English")
    inputData = os.listdir(inputData_folder+"raw_audio/")
    if output_folder is None:
        output_folder = inputData_folder+"transcript/"
        os.makedirs(output_folder, exist_ok=True)
    for data in inputData:
        input = inputData_folder+"raw_audio/"+data
        result = s2tpipe(input, generate_kwargs={"language": "english"})
        # result = model.transcribe(input,logprob_threshold=-1.0, compression_ratio_threshold=1.35)
        sound = AudioSegment.from_file(input, format="mp3")
        silence_ts = detect_silence(sound,
                                # split on silences longer than 1000ms (1 sec)
                                min_silence_len=2000,
                                # consider it silent if quieter than -16 dBFS
                                silence_thresh=-50)
        result["silence"] = (np.asarray(silence_ts)/1000).tolist()
        transcript = data.split(".")[0]+".json"
        with open(output_folder+transcript, 'w') as f:
            json.dump(result, f)
            
def split_onsilence_and_add(inputData_folder, output_format = "mp3", output_folder=None):
    inputData = os.listdir(inputData_folder+"raw_audio/")
    if output_folder is None:
        output_folder = inputData_folder+"processed_audio/"
        os.makedirs(output_folder, exist_ok=True)
    for data in inputData:
        sound = AudioSegment.from_file(inputData_folder+"raw_audio/"+data, format="mp3")
        # print(sound.duration_seconds)
        name, data_format = (data.split("."))
        # non_silent_list = detect_nonsilent(sound, silence_thresh = -23)
        # print(non_silent_list)
        chunks = split_on_silence(sound,
                                # split on silences longer than 1000ms (1 sec)
                                min_silence_len=2000,
                                # consider it silent if quieter than -50 dBFS
                                silence_thresh=-50,
                                # keep 200 ms of leading/trailing silence
                                keep_silence=1000)
        non_silent_sound = chunks[0]
        for i, chunk in enumerate(chunks):
            if i>0:
                non_silent_sound += chunk
        non_silent_sound.export(output_folder+name+"."+output_format,output_format)
        
if __name__ =="__main__":
    # split_onsilence_and_add("Surgical_videos_example/")
    # extract_audio("labeled_surgical_videos/")
    extract_text_from_mp3("labeled_surgical_videos/")

