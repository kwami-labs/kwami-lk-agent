"""Unit tests for Kwami configuration."""

import unittest

from src.domain import KwamiConfig, get_preset_config
from src.constants import LLMProviders, OpenAIModels, STTProviders, TTSProviders


class TestKwamiConfig(unittest.TestCase):
    def test_default_config(self):
        """Test default configuration values."""
        config = KwamiConfig()

        self.assertEqual(config.kwami_name, "Kwami")

        # Test voice defaults
        self.assertEqual(config.voice.pipeline_type, "standard")
        self.assertEqual(config.voice.stt_provider, STTProviders.DEEPGRAM)
        self.assertEqual(config.voice.llm_provider, LLMProviders.OPENAI)
        self.assertEqual(config.voice.tts_provider, TTSProviders.OPENAI)

        # Test memory defaults
        self.assertEqual(config.memory.enabled, False)  # Assuming no env var in test env

    def test_voice_presets(self):
        """Test voice preset configurations."""

        # Fast preset
        fast_config = get_preset_config("fast")
        self.assertEqual(fast_config.stt_provider, STTProviders.DEEPGRAM)
        self.assertEqual(fast_config.llm_provider, LLMProviders.GROQ)
        self.assertEqual(fast_config.tts_provider, TTSProviders.OPENAI)

        # Balanced preset
        balanced_config = get_preset_config("balanced")
        self.assertEqual(balanced_config.llm_provider, LLMProviders.OPENAI)
        self.assertEqual(balanced_config.llm_model, OpenAIModels.GPT_4O_MINI)

        # Quality preset
        quality_config = get_preset_config("quality")
        self.assertEqual(quality_config.llm_model, OpenAIModels.GPT_4O)
        self.assertEqual(quality_config.tts_provider, TTSProviders.ELEVENLABS)

        # Realtime preset
        realtime_config = get_preset_config("realtime")
        self.assertEqual(realtime_config.pipeline_type, "realtime")
        self.assertEqual(realtime_config.realtime_provider, LLMProviders.OPENAI)


if __name__ == "__main__":
    unittest.main()
