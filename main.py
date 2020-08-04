import pandas as pd
from pathlib import Path
from audio import AudioFeature

def parse_metadata(path):
    meta_df = pd.read_csv(path)
    meta_df = meta_df[["slice_file_name", "fold", "class"]]
    meta = zip(meta_df["slice_file_name"], meta_df["fold"], meta_df["class"])

    return meta

if __name__ == "__main__":
    metadata = parse_metadata("metadata/UrbanSound8K.csv")
    
    audio_features = []
    for row in metadata:
        path, fold, label = row
        src_path = f"{Path.home()}/datasets/UrbanSound8K/audio/fold{fold}/{path}"
        # TODO - add check; if on disk, load, if not, extract
        audio = AudioFeature(src_path, fold, label)
        audio.extract_features("mfcc", "spectral", "chroma")
        audio_features.append(audio)



