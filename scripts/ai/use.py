from TTS.api import TTS

tts = TTS(
    model_path='../training/gpt2/best_model_445860.pth',
    config_path='../training/gpt2/config.json'
).to('cuda')

tts.tts_to_file('Happy birthday!', file_path='wava.wav')
