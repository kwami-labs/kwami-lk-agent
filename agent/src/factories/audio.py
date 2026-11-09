"""Resampler warm-up.

`livekit_ffi.dll` bundles libsoxr built without OpenMP, which leaves soxr's FFT
cache (`fft4g_cache.h`) with no locking at all, and with its asserts compiled
in. The cache is initialised lazily on first use and grown whenever a resampler
needs a longer FFT than any before it. A session starts several resamplers at
once (room input, STT, VAD, TTS output), so two threads can hit that first
initialisation together; one then trips `assert(LSX_FFT_BR == NULL)` and the
Visual C++ runtime aborts the whole process -- on Windows, as a modal dialog.

The cache never shrinks, so running every rate/quality the pipeline uses once,
on one thread, before any session starts leaves nothing for those threads to
initialise or grow. It has to run in every job process: each one loads its own
copy of the DLL and therefore its own cache.
"""

from __future__ import annotations

from livekit import rtc

from ..utils.logging import get_logger

logger = get_logger("factories")

#: Room audio, STT/VAD input, realtime models and the usual TTS output rates.
_RATES = (8000, 16000, 22050, 24000, 44100, 48000)


def prewarm_resampler() -> None:
    """Initialise soxr's FFT cache for every rate pair and quality, single-threaded."""
    count = 0
    for quality in rtc.AudioResamplerQuality:
        for input_rate in _RATES:
            for output_rate in _RATES:
                if input_rate == output_rate:
                    continue
                resampler = rtc.AudioResampler(input_rate, output_rate, quality=quality)
                # 100 ms of silence is enough for soxr to build its filters.
                resampler.push(bytearray(input_rate // 10 * 2))
                resampler.flush()
                count += 1
    logger.debug("Prewarmed %d soxr resamplers", count)
