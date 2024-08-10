class NoVoiceLines(Exception):
    """Raised when ServantVoices used without initiating via buildVoiceLinesDict()"""

class NoSuchCategory(Exception):
    """Raised when ServantVoices used without initiating via buildVoiceLinesDict()"""

class FFMPEGException(Exception):
    """Raised when FFMpeg launch returns non-zero code (when some fault occurred)"""

class DownloadException(Exception):
    """Raised when downloading file impossible or error occurred during download"""
