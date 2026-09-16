"""Unit tests for factory functions."""

import unittest
from unittest.mock import patch

# Note: livekit mocking is done in conftest.py
from src.config import KwamiVoiceConfig
from src.constants import OpenAIModels, OpenAIVoices, STTProviders, TTSProviders
from src.factories.stt import create_stt
from src.factories.tts import create_tts


class TestFactories(unittest.TestCase):
    @patch("src.factories.tts.openai.TTS")
    @patch("os.getenv")
    def test_create_openai_tts(self, mock_getenv, mock_openai_tts):
        """Test creating OpenAI TTS."""
        # Mock env var to avoid warning
        mock_getenv.return_value = "fake-key"

        config = KwamiVoiceConfig(
            tts_provider=TTSProviders.OPENAI,
            tts_model=OpenAIModels.TTS_1,
            tts_voice=OpenAIVoices.NOVA,
            tts_speed=1.2,
        )

        create_tts(config)

        mock_openai_tts.assert_called_once_with(
            model=OpenAIModels.TTS_1, voice=OpenAIVoices.NOVA, speed=1.2
        )

    @patch("src.factories.tts.openai.TTS")
    @patch("os.getenv")
    def test_create_openai_tts_fallback(self, mock_getenv, mock_openai_tts):
        """Test OpenAI TTS fallback for invalid voice/model."""
        # Mock env var to avoid warning
        mock_getenv.return_value = "fake-key"

        config = KwamiVoiceConfig(
            tts_provider=TTSProviders.OPENAI, tts_model="invalid-model", tts_voice="invalid_voice"
        )

        create_tts(config)

        # Should fallback to defaults
        mock_openai_tts.assert_called_once_with(
            model=OpenAIModels.TTS_1, voice=OpenAIVoices.DEFAULT, speed=1.0
        )

    @patch("src.factories.stt.deepgram.STT")
    def test_create_deepgram_stt(self, mock_deepgram_stt):
        """Test creating Deepgram STT."""
        config = KwamiVoiceConfig(
            stt_provider=STTProviders.DEEPGRAM, stt_model="nova-2-medical", stt_language="fr"
        )

        create_stt(config)

        mock_deepgram_stt.assert_called_once()
        call_args = mock_deepgram_stt.call_args[1]
        self.assertEqual(call_args["model"], "nova-2-medical")
        self.assertEqual(call_args["language"], "fr")


if __name__ == "__main__":
    unittest.main()
