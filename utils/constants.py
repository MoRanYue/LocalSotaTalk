"""TTS后端服务常量定义"""

# 支持的语言列表
# 根据两个框架的能力定义支持的语言
SUPPORTED_LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish", 
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "pt": "Portuguese",
    "hi": "Hindi",
    "bn": "Bengali",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "pl": "Polish",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "el": "Greek",
    "hu": "Hungarian",
    "cs": "Czech",
    "ro": "Romanian",
    "uk": "Ukrainian",
    "id": "Indonesian",
    "ms": "Malay",
    "fil": "Filipino",
    "he": "Hebrew",
    "fa": "Persian",
    "ur": "Urdu",
}

# OmniVoice支持600+语言，这里列出常用语言
# 如果需要完整列表，可以动态从框架获取
OMNIVOICE_LANGUAGES = {
    **SUPPORTED_LANGUAGES,
    # 添加更多OmniVoice支持的语言
    "af": "Afrikaans",
    "am": "Amharic", 
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "cy": "Welsh",
    "et": "Estonian",
    "eu": "Basque",
    "ga": "Irish",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "hr": "Croatian",
    "hy": "Armenian",
    "is": "Icelandic",
    "jv": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ky": "Kyrgyz",
    "la": "Latin",
    "lb": "Luxembourgish",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "mt": "Maltese",
    "my": "Burmese",
    "ne": "Nepali",
    "ps": "Pashto",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "su": "Sundanese",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "tl": "Tagalog",
    "tt": "Tatar",
    "ug": "Uyghur",
    "uz": "Uzbek",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zu": "Zulu",
}

# LongCat-AudioDiT支持的语言
LONGCAT_LANGUAGES = {
    "en": "English",
    "zh": "Chinese",
}

# 说话人类型
SPEAKER_TYPES = {
    "audio_only": "仅音频",
    "audio_with_text": "音频+文本",
    "design_only": "仅设计",
}

# 默认TTS设置
DEFAULT_TTS_SETTINGS = {
    "stream_chunk_size": 1024,
    "temperature": 1.0,
    "speed": 1.0,
    "length_penalty": 1.0,
    "repetition_penalty": 1.0,
    "top_p": 0.9,
    "top_k": 50,
    "enable_text_splitting": True,
}

# 音频格式
AUDIO_FORMATS = {
    "sample_rate": 24000,  # 两个框架都使用24kHz
    "format": "wav",
    "bits_per_sample": 16,
}