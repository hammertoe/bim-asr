# Data

This document lists every dataset Bim-ASR touches, where it comes from,
and what its licence allows. We are deliberately conservative on
licences because Bim-ASR will be redistributed.

## Datasets we use

### LibriSpeech (read English, public)

- Source: [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr)
- Variants: train-clean-100, train-clean-360, dev-clean, dev-other, test-clean, test-other
- Licence: CC-BY-4.0 (audio and transcripts)
- Size: 1000 hours total; train-clean-100 is 100 hours
- Use in Bim-ASR: stages 1-2 (training), stage 5 (evaluation)
- Notes: read speech from public-domain audiobooks. High quality, low
  noise, no Barbadian content. Good for proving the SALM scaffolding
  works.

### GigaSpeech (broad-domain English, gated)

- Source: [speechcolab/gigaspeech](https://huggingface.co/datasets/speechcolab/gigaspeech)
- Variants: XS (10 h), S (250 h), M (1000 h), L (2500 h), XL (10 000 h)
- Licence: research-only via gated access; AudioSet-derived audio is
  CC-BY, audiobook and podcast audio is CC-BY-SA. **Cannot redistribute
  the audio or transcripts as part of the model card.**
- Size: XS is 10 hours, S is 250 hours
- Use in Bim-ASR: stage 2 training, stage 5 evaluation
- Notes: covers audiobook, podcast, and YouTube. More diverse than
  LibriSpeech, more like real broadcast audio. Need to fill out the
  access form before downloading.

### Common Voice 17 (multilingual, public)

- Source: [mozilla-foundation/common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)
- Variants: per-language splits, with a 1.6 GB English subset
- Licence: CC-0
- Size: thousands of hours across all languages; English alone is
  roughly 1 600 hours
- Use in Bim-ASR: stage 3 Caribbean training, stage 5 Caribbean
  evaluation
- Notes: includes Caribbean English speakers (Jamaica, Trinidad, Barbados,
  Guyana, Belize). We will filter on the speaker metadata. This is
  the first source of real Caribbean-accented audio.

### Whisper-large-v3 frontier transcripts (Barbados Parliament)

- Source: locally produced. We run
  [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
  on Barbados Parliament YouTube audio, then hand-correct the
  transcripts to make a paired dataset.
- Audio source: the
  [Barbados Parliament Channel](https://www.youtube.com/channel/UCnAo0XVbxFF2UCpkIkAS-6A)
  and the
  [Vimeo Parliament showcase](https://vimeo.com/showcase/6925951).
- Licence: Parliament audio is public (broadcast by the Parliament
  itself). Transcripts we produce are CC-BY-4.0.
- Size: we start with 5-10 hours of paired audio.
- Use in Bim-ASR: stage 4 training, stage 5 evaluation
- Notes: the pairing quality matters more than the quantity. We will
  pick Parliament sessions with good audio (no music beds, no
  cross-talk) and produce word-level alignment with the NeMo Forced
  Aligner.

## Datasets we deliberately do not use

- **MLS (Multilingual LibriSpeech).** Bigger than LibriSpeech but
  derived from the same LibriVox pool. Skipping to keep the licence
  story simple.
- **People's Speech.** Large, but the licence is unclear for
  redistribution.
- **VoxPopuli, MuST-C.** Multilingual, but the focus is European
  parliament speech. We want Caribbean, not European.
- **TED-LIUM.** Clean TED talks. Nice to have, but adds little for
  Barbados.

## How we keep the data clean

For each training clip we store:

- `audio_filepath`: absolute path
- `duration`: float seconds
- `text`: clean transcript
- `source_dataset`: where the clip came from
- `source_split`: train/dev/test
- `language`: BCP-47 tag, default `en-BB` for Barbados
- `speaker_id`: where available (Common Voice, Parliament)
- `whisper_logprob`: mean log-probability of the transcript under
  Whisper-large-v3, used to drop low-confidence clips

We also store the SHA-256 of every audio file and every transcript
line, so we can detect duplicates and re-runs.

## How we prepare the NeMo manifest

NeMo SALM training expects a JSONL manifest with at least
`audio_filepath`, `duration`, and `text`. We add the metadata above
as extra fields and let NeMo pick out what it needs.

Example manifest line:

    {"audio_filepath": "/data/cv17/en/jb_001.wav",
     "duration": 7.42,
     "text": "I went down to Oistins on Saturday morning.",
     "source_dataset": "common_voice_17",
     "source_split": "train",
     "language": "en-BB",
     "speaker_id": "cv_jb_001",
     "whisper_logprob": -0.21}

## Downloading the data

We use `huggingface-cli` for the public datasets and a manual
download for GigaSpeech. The download script lives in
`scripts/download_data.py`.

- LibriSpeech: `huggingface-cli download openslr/librispeech_asr --repo-type dataset --include "train.clean.100/*"`
- Common Voice 17: `huggingface-cli download mozilla-foundation/common_voice_17_0 --repo-type dataset --include "en/*"`
- GigaSpeech: sign in, accept the gate, then `huggingface-cli download speechcolab/gigaspeech --repo-type dataset --include "xs/*"`
- Parliament audio: `yt-dlp` against the YouTube and Vimeo URLs.

## Privacy and consent

Parliament audio is public broadcast and is acceptable to use. We will
not transcribe any private recordings, telephone calls, or anything
where a person could have a reasonable expectation of privacy. Common
Voice contributors consented to CC-0 release. We are not adding any
new personally identifying information.

## Versioning

The data version is recorded in
`data/MANIFEST_VERSION`. We pin the Hugging Face dataset revisions
in `data/locks.json` so that re-runs hit the same audio bytes.
