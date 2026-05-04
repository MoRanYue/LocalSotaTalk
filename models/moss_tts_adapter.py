"""MOSS-TTS 统一适配器

支持 MOSS-TTS 家族全部架构：
  - MossTTSDelay   (MOSS-TTS 8B, MOSS-TTSD, MOSS-VoiceGenerator, MOSS-SoundEffect)
  - MossTTSLocal   (MOSS-TTS-Local-Transformer 1.7B)
  - MossTTSRealtime (MOSS-TTS-Realtime 1.7B)

直接使用 systems/MOSS-TTS/ submodule 中的本地建模代码，
不依赖 HuggingFace Hub 的 trust_remote_code 动态加载机制。
"""
import sys
import importlib.util
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

from models.base_adapter import BaseTTSAdapter
from utils.constants import MOSS_TTS_LANGUAGES, DEFAULT_TTS_SETTINGS

# 将 systems/MOSS-TTS 加入 sys.path，使本地子模块可被导入
_LOCAL_MOSS_TTS_PATH = Path(__file__).parent.parent / "systems" / "MOSS-TTS"
if _LOCAL_MOSS_TTS_PATH.exists():
    sys.path.insert(0, str(_LOCAL_MOSS_TTS_PATH))
    # 同时把 moss_tts_realtime 子模块也加入 path
    _REALTIME_PATH = _LOCAL_MOSS_TTS_PATH / "moss_tts_realtime"
    if _REALTIME_PATH.exists() and str(_REALTIME_PATH) not in sys.path:
        sys.path.insert(0, str(_REALTIME_PATH))

# ---- 预导入本地建模代码，注册 model_type → class 映射 ----
def _pre_import_moss_tts_local_code() -> List[str]:
    """导入本地 MOSS-TTS 建模代码，使 AutoModel / AutoProcessor 能识别这些架构。"""
    imported: List[str] = []
    # MossTTSDelay (8B, VoiceGenerator, SoundEffect 等)
    try:
        from moss_tts_delay.configuration_moss_tts import MossTTSDelayConfig  # noqa: F401
        from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel       # noqa: F401
        imported.append("moss_tts_delay")
    except ImportError as e:
        print(f"[MOSS-TTS] Warning: Could not import moss_tts_delay: {e}")
    # MossTTSLocal (1.7B Local Transformer — 与 Delay 共享 model_type)
    try:
        from moss_tts_local.configuration_moss_tts import MossTTSDelayConfig  # noqa: F401
        from moss_tts_local.modeling_moss_tts import MossTTSDelayModel       # noqa: F401
        imported.append("moss_tts_local")
    except ImportError as e:
        print(f"[MOSS-TTS] Warning: Could not import moss_tts_local: {e}")
    # MossTTSRealtime (1.7B Realtime)
    try:
        from mossttsrealtime.configuration_mossttsrealtime import MossTTSRealtimeConfig  # noqa: F401
        from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime              # noqa: F401
        imported.append("mossttsrealtime")
    except ImportError as e:
        print(f"[MOSS-TTS] Warning: Could not import mossttsrealtime: {e}")
    if not imported:
        print("[MOSS-TTS] Warning: No local MOSS-TTS modeling modules could be imported.")
    else:
        print(f"[MOSS-TTS] Pre-imported local modeling modules: {', '.join(imported)}")
    return imported


_PRE_IMPORTED = _pre_import_moss_tts_local_code()


class MossTTSAdapter(BaseTTSAdapter):
    """MOSS-TTS 统一适配器，直接使用本地建模代码加载模型。"""

    def __init__(self, model_repo: str, device: str = "auto"):
        super().__init__(model_repo, device=device)
        self.processor = None           # Processor 实例（延迟或实时模式类型不同）
        self.audio_tokenizer = None     # processor.audio_tokenizer
        self.model_config = None        # processor.model_config
        self.generation_config: Dict[str, Any] = {}
        self.sample_rate = 24000        # MOSS-TTS 默认 24 kHz
        self._model_device: Optional[str] = None
        self._model_dtype: Optional[torch.dtype] = None
        self._model_type: Optional[str] = None      # 模型类型字符串
        self._is_realtime: bool = False             # 是否为 Realtime 架构
        self._codec = None                          # Realtime 架构专用的 codec 模型
        self._inferencer = None                     # Realtime 架构专用的推理引擎

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def load_model(self):
        """加载 MOSS-TTS 模型。

        优先使用 AutoModel / AutoProcessor（若本地模块已注册映射），
        若失败则回退到直接使用本地建模类构造。
        """
        try:
            from transformers import AutoModel, AutoProcessor  # type: ignore
        except ImportError:
            raise ImportError(
                "transformers is required for MOSS-TTS. "
                "Install via: pip install transformers>=5.0.0"
            )

        device = self.device
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._model_device = device
        self._model_dtype = dtype

        # 禁用 cuDNN SDPA（MOSS-TTS 官方建议）
        if device == "cuda":
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

        attn_implementation = self._resolve_attn_implementation(device, dtype)

        print(f"[MOSS-TTS] Loading model from {self.model_repo} ...")
        print(f"[MOSS-TTS] device={device}, dtype={dtype}, attn={attn_implementation}")

        # 0. 确定模型类型
        self._model_type = self._resolve_raw_model_type(self.model_repo)
        self._is_realtime = "moss_tts_realtime" in (self._model_type or "")

        # 1. 加载 Processor（对 Realtime 架构需要特殊处理）
        self._load_processor()

        # 2. 将 audio_tokenizer 移至目标设备
        if hasattr(self.processor, "audio_tokenizer") and self.processor.audio_tokenizer is not None:
            self.audio_tokenizer = self.processor.audio_tokenizer.to(device)
        else:
            self.audio_tokenizer = None

        # 3. 加载模型 — 先尝试 AutoModel，失败则回退到直接实例化
        try:
            self.model = AutoModel.from_pretrained(
                self.model_repo,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                torch_dtype=dtype,
            ).to(device)
        except (ValueError, KeyError, RuntimeError) as e:
            print(f"[MOSS-TTS] AutoModel.from_pretrained failed ({e}), "
                  f"falling back to direct class instantiation...")
            self.model = self._load_model_direct(dtype).to(device=device, dtype=dtype)

        self.model.eval()

        # 4. 获取 model_config（提供 sampling_rate 等信息）
        if hasattr(self.processor, "model_config"):
            self.model_config = self.processor.model_config
            if hasattr(self.model_config, "sampling_rate"):
                self.sample_rate = int(self.model_config.sampling_rate)

        # 5. 默认生成配置
        self.generation_config = {
            "text_temperature": 0.9,
            "text_top_p": 0.9,
            "text_top_k": 50,
            "audio_temperature": 0.9,
            "audio_top_p": 0.9,
            "audio_top_k": 50,
            "audio_repetition_penalty": 1.0,
            "max_new_tokens": 4096,
            "n_vq_for_inference": None,
        }

        # 6. 对 Realtime 架构：加载 codec 模型并构建推理引擎
        if self._is_realtime:
            self._load_codec()

        self.is_loaded = True
        print(f"[MOSS-TTS] Model loaded successfully. sample_rate={self.sample_rate}")

    def _load_processor(self):
        """加载 Processor，对 Realtime 架构绕开 AutoProcessor 的类型解析故障。"""
        if not self._is_realtime:
            # Delay/Local 架构：AutoProcessor 可正常处理
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                self.model_repo,
                trust_remote_code=True,
            )
        else:
            # Realtime 架构：AutoProcessor 回退到错误的处理器，
            # 需要手动构建 MossTTSRealtimeProcessor(tokenizer)
            print("[MOSS-TTS] Realtime model detected: loading processor manually...")
            from transformers import AutoTokenizer
            from mossttsrealtime.processing_mossttsrealtime import MossTTSRealtimeProcessor

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_repo,
                trust_remote_code=True,
            )
            self.processor = MossTTSRealtimeProcessor(tokenizer=tokenizer)

    def _load_codec(self):
        """加载 Realtime 架构需要的音频编解码模型。"""
        codec_repo = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
        try:
            from transformers import AutoModel
            device = self._model_device or "cpu"
            print(f"[MOSS-TTS] Loading codec from {codec_repo} ...")
            self._codec = AutoModel.from_pretrained(
                codec_repo,
                trust_remote_code=True,
            ).eval().to(device)
            print(f"[MOSS-TTS] Codec loaded on {device}")
        except Exception as e:
            print(f"[MOSS-TTS] Warning: Failed to load codec ({e}). "
                  f"Realtime TTS synthesis will not be available.")

    def _load_model_direct(self, torch_dtype: torch.dtype):
        """直接使用本地建模类加载模型（绕过 AutoModel 的 model_type 查找）。

        从模型仓库的 config.json 中解析 model_type，
        然后手动实例化正确的配置类，再使用 from_pretrained(config=config) 
        加载预训练权重（避免 from_pretrained 内部依赖 AutoConfig）。
        """
        model_path = self.model_repo
        model_type = self._resolve_raw_model_type(model_path)

        if "moss_tts_realtime" in model_type:
            from mossttsrealtime.configuration_mossttsrealtime import MossTTSRealtimeConfig
            from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
            # 先手动加载正确的配置
            config = MossTTSRealtimeConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            # 使用 config=config 绕过 from_pretrained 内部的 AutoConfig
            model = MossTTSRealtime.from_pretrained(
                model_path,
                config=config,
                trust_remote_code=False,  # 配置已手动加载，不再需要 trust_remote_code
                torch_dtype=torch_dtype,
            )
        else:
            # moss_tts_delay（包括 MOSS-TTSD, VoiceGenerator, SoundEffect）
            from moss_tts_delay.configuration_moss_tts import MossTTSDelayConfig
            from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel
            # 先手动加载正确的配置
            config = MossTTSDelayConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            model = MossTTSDelayModel.from_pretrained(
                model_path,
                config=config,
                trust_remote_code=False,
                torch_dtype=torch_dtype,
            )

        return model

    @staticmethod
    def _resolve_raw_model_type(model_path: str) -> str:
        """直接读取 config.json 的原始内容以获取 model_type，
        避免依赖 AutoConfig 的类型解析（可能在本地模块未完全注册时返回错误类型）。
        """
        import json

        local_path = Path(model_path)
        cfg_path = local_path / "config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg.get("model_type", "")

        # 远程 HuggingFace Hub — 下载 config.json
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(repo_id=model_path, filename="config.json")
            with open(downloaded, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg.get("model_type", "")
        except ImportError:
            pass
        except Exception:
            pass

        # 最后的回退
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        return getattr(config, "model_type", "")

    # ------------------------------------------------------------------
    # 合成
    # ------------------------------------------------------------------
    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> np.ndarray:
        """
        合成语音（支持零样本 + 语音克隆）

        Args:
            text: 要合成的文本
            speaker_wav: 说话人参考音频路径（可选，用于语音克隆）
            language: 语言代码
            **kwargs: 生成参数覆盖

        Returns:
            np.ndarray: 单声道音频数据
        """
        self.ensure_loaded()
        return self._do_synthesize(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            instruction=None,
            **kwargs,
        )

    def synthesize_instructively(
        self,
        text: str,
        design_description: str,
        language: str = "en",
        **kwargs
    ) -> np.ndarray:
        """
        通过音频设计描述合成语音（MOSS-VoiceGenerator 等模型支持）

        Args:
            text: 要合成的文本
            design_description: 音频设计描述文本
            language: 语言代码
            **kwargs: 生成参数覆盖

        Returns:
            np.ndarray: 单声道音频数据
        """
        self.ensure_loaded()
        return self._do_synthesize(
            text=text,
            speaker_wav=None,
            language=language,
            instruction=design_description,
            **kwargs,
        )

    def _do_synthesize(
        self,
        text: str,
        speaker_wav: Optional[str],
        language: str,
        instruction: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """核心合成逻辑。根据模型类型分发到不同的合成路径。"""
        if self._is_realtime:
            return self._do_synthesize_realtime(
                text=text,
                speaker_wav=speaker_wav,
                **kwargs,
            )
        else:
            return self._do_synthesize_delay(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                instruction=instruction,
                **kwargs,
            )

    def _do_synthesize_realtime(
        self,
        text: str,
        speaker_wav: Optional[str],
        **kwargs,
    ) -> np.ndarray:
        """Realtime 架构的合成逻辑。

        使用 MossTTSRealtimeInference + 外部 Codec 解码。
        """
        if self._codec is None:
            # 尝试再次加载 codec
            self._load_codec()
        if self._codec is None:
            raise RuntimeError(
                "Codec model is required for Realtime TTS synthesis but failed to load. "
                "Please ensure HuggingFace can access 'OpenMOSS-Team/MOSS-Audio-Tokenizer'."
            )

        # 合并生成参数
        gen_kwargs = {**self.generation_config, **kwargs}

        from inferencer import MossTTSRealtimeInference

        # 确保有 或创建 推理引擎
        if self._inferencer is None:
            device = torch.device(self._model_device or "cpu")
            self._inferencer = MossTTSRealtimeInference(
                model=self.model,
                tokenizer=self.processor.tokenizer,
                max_length=int(gen_kwargs.get("max_new_tokens", 4096)),
                codec=self._codec,
                codec_sample_rate=self.sample_rate,
                codec_encode_kwargs={"chunk_duration": 8},
            )

        # 执行生成
        device = torch.device(self._model_device or "cpu")
        result_tokens_list = self._inferencer.generate(
            text=text,
            reference_audio_path=speaker_wav if speaker_wav and Path(speaker_wav).exists() else None,
            max_length=int(gen_kwargs.get("max_new_tokens", 4096)),
            temperature=float(gen_kwargs.get("audio_temperature", 0.9)),
            top_p=float(gen_kwargs.get("audio_top_p", 0.9)),
            top_k=int(gen_kwargs.get("audio_top_k", 50)),
            do_sample=True,
            repetition_penalty=float(gen_kwargs.get("audio_repetition_penalty", 1.0)),
            repetition_window=50,
            device=device,
        )

        # 第一个结果的 tokens（batch=1 场景）
        tokens = result_tokens_list[0]  # shape: [T, C]

        # 用 codec 解码为波形
        tokens_tensor = torch.from_numpy(tokens).to(device)
        # codec.decode 期望 [C, T] 格式
        decode_result = self._codec.decode(tokens_tensor.permute(1, 0), chunk_duration=8)
        wav = decode_result["audio"][0].cpu().detach().numpy()

        if wav.ndim == 1:
            return wav
        return wav.squeeze()

    def _do_synthesize_delay(
        self,
        text: str,
        speaker_wav: Optional[str],
        language: str,
        instruction: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """Delay/Local 架构的合成逻辑。"""
        # 合并生成参数
        gen_kwargs = {**self.generation_config, **kwargs}

        # 构建会话消息
        conversation = self._build_conversation(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            instruction=instruction,
            **gen_kwargs,
        )

        # 预处理
        batch = self.processor([conversation], mode="generation")
        input_ids = batch["input_ids"].to(self._model_device)
        attention_mask = batch["attention_mask"].to(self._model_device)

        # 提取 model.generate 接受的参数
        generate_params = self._prepare_generate_params(gen_kwargs)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_params,
            )

        # 解码
        messages = self.processor.decode(outputs)
        audio = messages[0].audio_codes_list[0]

        # 转为 numpy
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = np.array(audio)

        if len(audio_np.shape) > 1:
            audio_np = audio_np.squeeze()

        return audio_np

    # ------------------------------------------------------------------
    # 消息构建（仅 Delay/Local 架构使用）
    # ------------------------------------------------------------------
    def _build_conversation(
        self,
        text: str,
        speaker_wav: Optional[str],
        language: str,
        instruction: Optional[str],
        **gen_kwargs
    ) -> List[Dict[str, Any]]:
        """构建 processor.build_user_message() 所用参数列表。"""
        msg_kwargs: Dict[str, Any] = {"text": text}

        if language:
            msg_kwargs["language"] = language

        if speaker_wav:
            ref_path = Path(speaker_wav)
            if ref_path.suffix == ".design.txt":
                with open(speaker_wav, "r", encoding="utf-8") as f:
                    instruction = f.read().strip()
            elif ref_path.exists():
                msg_kwargs["reference"] = [str(ref_path)]

        if instruction:
            msg_kwargs["instruction"] = instruction

        if "tokens" in gen_kwargs:
            msg_kwargs["tokens"] = gen_kwargs["tokens"]

        return [self.processor.build_user_message(**msg_kwargs)]

    # ------------------------------------------------------------------
    # 生成参数
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_generate_params(gen_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """从 gen_kwargs 中提取 model.generate() 接受的参数。"""
        supported = {
            "max_new_tokens",
            "temperature",
            "text_temperature",
            "text_top_p",
            "text_top_k",
            "audio_temperature",
            "audio_top_p",
            "audio_top_k",
            "audio_repetition_penalty",
            "top_p",
            "top_k",
            "repetition_penalty",
            "do_sample",
            "n_vq_for_inference",
        }
        return {k: v for k, v in gen_kwargs.items() if k in supported and v is not None}

    # ------------------------------------------------------------------
    # Attention 实现选择
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_attn_implementation(device: str, dtype: torch.dtype) -> str:
        """选择合适的 attention 后端。"""
        if device == "cuda" and importlib.util.find_spec("flash_attn") is not None \
                and dtype in (torch.float16, torch.bfloat16):
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return "flash_attention_2"
        if device == "cuda":
            return "sdpa"
        return "eager"

    # ------------------------------------------------------------------
    # 接口实现
    # ------------------------------------------------------------------
    def get_supported_languages(self) -> Dict[str, str]:
        return MOSS_TTS_LANGUAGES.copy()

    def get_tts_settings(self) -> Dict[str, Any]:
        settings = DEFAULT_TTS_SETTINGS.copy()
        settings.update(self.generation_config)
        return settings

    def update_tts_settings(self, settings: Dict[str, Any]):
        valid_keys = {
            "text_temperature", "text_top_p", "text_top_k",
            "audio_temperature", "audio_top_p", "audio_top_k",
            "audio_repetition_penalty", "max_new_tokens",
            "n_vq_for_inference", "tokens",
        }
        for k in valid_keys:
            if k in settings:
                self.generation_config[k] = settings[k]
        for k, v in settings.items():
            if k not in valid_keys:
                self.generation_config[k] = v

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "framework": "mosstts",
            "model_repo": self.model_repo,
            "sample_rate": self.sample_rate,
            "generation_config": self.generation_config.copy(),
            "model_type": self._model_type,
            "is_realtime": self._is_realtime,
        }

    def cleanup(self):
        """释放 GPU 资源。"""
        self.model = None
        self.processor = None
        self.audio_tokenizer = None
        self.model_config = None
        self._codec = None
        self._inferencer = None
        self.is_loaded = False
        self._model_device = None
        self._model_dtype = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        self.cleanup()