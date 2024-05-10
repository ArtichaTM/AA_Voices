from pathlib import Path

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.delightful_tts_config import DelightfulTtsAudioConfig, DelightfulTTSConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.delightful_tts import DelightfulTTS, DelightfulTtsArgs, VocoderConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio.processor import AudioProcessor

ROOT_PATH = Path(__file__) / '..' / '..' / '..'
OUTPUT_PATH = ROOT_PATH / 'training'
DATA_PATH = ROOT_PATH / 'dataset' / 'vctk'
F0_CACHE_PATH = OUTPUT_PATH / 'f0_cache'
PHONEME_CACHE_PATH = OUTPUT_PATH / 'phoneme_cache'

dataset_config = BaseDatasetConfig(
    dataset_name="vctk", formatter="vctk", meta_file_train="", path=str(DATA_PATH.resolve()), language="en-us"
)

audio_config = DelightfulTtsAudioConfig()

model_args = DelightfulTtsArgs()

vocoder_config = VocoderConfig()

something_tts_config = DelightfulTTSConfig(
    run_name="delightful_tts_vctk",
    run_description="Train like in delightful tts paper.",
    model_args=model_args,
    audio=audio_config,
    vocoder=vocoder_config,
    batch_size=16,
    eval_batch_size=16,
    num_loader_workers=10,
    num_eval_loader_workers=10,
    precompute_num_workers=3,
    compute_input_seq_cache=True,
    compute_f0=True,
    f0_cache_path=str(F0_CACHE_PATH.resolve()),
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=str(PHONEME_CACHE_PATH.resolve()),
    print_step=50,
    print_eval=False,
    mixed_precision=True,
    output_path=str(OUTPUT_PATH.resolve()),
    datasets=[dataset_config],
    start_by_longest=True,
    binary_align_loss_alpha=0.0,
    use_attn_priors=False,
    max_text_len=80,
    steps_to_start_discriminator=10000,
)

tokenizer, config = TTSTokenizer.init_from_config(something_tts_config)

ap = AudioProcessor.init_from_config(config)


train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)


speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers


model = DelightfulTTS(ap=ap, config=config, tokenizer=tokenizer, speaker_manager=speaker_manager)

trainer = Trainer(
    TrainerArgs(gpu=0), config, str(OUTPUT_PATH.resolve()), model=model, train_samples=train_samples, eval_samples=eval_samples
)
