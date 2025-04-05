# tts.py
# --------------------------------------------------------------------
# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
import httpx

# LiveKit imports
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)

# OpenAI plugin
import openai

# Local references (from same directory/package)
from .models import TTSModels, TTSVoices
from .utils import AsyncAzureADTokenProvider

# Audio constants
OPENAI_TTS_SAMPLE_RATE = 24000
OPENAI_TTS_CHANNELS = 1


@dataclass
class _TTSOptions:
    """
    Internal dataclass to store TTS options like model, voice, speed, etc.
    """
    model: TTSModels | str
    voice: TTSVoices | str
    speed: float


class TTS(tts.TTS):
    """
    Main TTS class implementing an OpenAI-compatible text-to-speech client.
    """

    def __init__(
        self,
        *,
        model: TTSModels | str = "tts-1",
        voice: TTSVoices | str = "alloy",
        speed: float = 1.0,
        base_url: str | None = None,
        api_key: str | None = None,
        client: openai.AsyncClient | None = None,
    ) -> None:
        """
        Create a new instance of OpenAI-compatible TTS.

        If no 'client' is provided, we'll build an openai.AsyncClient with the given
        base_url and api_key.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,  # Or True if you want to do streaming
            ),
            sample_rate=OPENAI_TTS_SAMPLE_RATE,
            num_channels=OPENAI_TTS_CHANNELS,
        )

        # Store our TTS options
        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            speed=speed,
        )

        # Build the openai.AsyncClient if not provided.
        # IMPORTANT: single default timeout to avoid ValueError.
        self._client = client or openai.AsyncClient(
            max_retries=0,
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(15.0),  # Single default for all phases
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )

    def update_options(
        self,
        *,
        model: TTSModels | None = None,
        voice: TTSVoices | None = None,
        speed: float | None = None,
    ) -> None:
        """
        Update runtime TTS options as needed (model, voice, speed, etc.).
        """
        self._opts.model = model or self._opts.model
        self._opts.voice = voice or self._opts.voice
        self._opts.speed = speed or self._opts.speed

    #
    # ---------------------------------------------------------------------
    #  Azure Client Example (if you need it, otherwise remove this section)
    # ---------------------------------------------------------------------
    #
    @staticmethod
    def create_azure_client(
        *,
        model: TTSModels = "tts-1",
        voice: TTSVoices = "alloy",
        speed: float = 1.0,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        azure_ad_token_provider: AsyncAzureADTokenProvider | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
    ) -> TTS:
        """
        Demonstration of how an Azure-based TTS client could be created.
        """
        azure_client = openai.AsyncAzureOpenAI(
            max_retries=0,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
            azure_ad_token_provider=azure_ad_token_provider,
            organization=organization,
            project=project,
            base_url=base_url,
        )  # type: ignore

        return TTS(
            model=model,
            voice=voice,
            speed=speed,
            client=azure_client,
        )

    #
    # ---------------------------------------------------------------------
    #        Kokoro Client Method - Single Timeout to Avoid ValueError
    # ---------------------------------------------------------------------
    #
    @staticmethod
    def create_kokoro_client(
        *,
        model: TTSModels | str = "kokoro",
        voice: TTSVoices | str = "af_sky",
        speed: float = 1.0,
        base_url: str = "http://localhost:8880/v1",
        api_key: str = "not-needed",
    ) -> TTS:
        """
        Create a TTS client pointing to a local Kokoro TTS endpoint,
        which is OpenAI-compatible.
        """
        kokoro_client = openai.AsyncClient(
            max_retries=0,
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(15.0),  # Single default
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )

        return TTS(
            model=model,
            voice=voice,
            speed=speed,
            client=kokoro_client,
        )

    #
    # ---------------------------------------------------------------------
    #        Orpheus Client Method - Single Timeout to Avoid ValueError
    # ---------------------------------------------------------------------
    #
    @staticmethod
    def create_orpheus_client(
        *,
        voice: TTSVoices | str = "tara",  # Default voice from Orpheus
        base_url: str = "http://localhost:5005/v1",
        api_key: str = "not-needed",
    ) -> TTS:
        """
        Create a TTS client pointing to a local Orpheus TTS endpoint,
        which is OpenAI-compatible.
        
        Available Orpheus voices:
        - tara - Best overall voice for general use (default)
        - leah
        - jess
        - leo
        - dan
        - mia
        - zac
        - zoe
        
        Parameters:
        -----------
        voice: TTSVoices | str
            One of the available Orpheus voices (defaults to "tara")
        base_url: str
            URL of the Orpheus TTS service (default: http://localhost:5005/v1)
        api_key: str
            API key (typically not needed for local installations)
            
        Returns:
        --------
        TTS
            Configured TTS instance for Orpheus
        """
        orpheus_client = openai.AsyncClient(
            max_retries=0,
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(15.0),  # Single default timeout
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )

        return TTS(
            model="not-needed",  # Model parameter is not used for Orpheus
            voice=voice,
            speed=1.0,  # Using default speed
            client=orpheus_client,
        )

    #
    # ---------------------------------------------------------------------
    #  Primary Synthesize method (returns a stream of audio chunks)
    # ---------------------------------------------------------------------
    #
    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "ChunkedStream":
        """
        Called by the VoicePipelineAgent to get a stream of synthesized audio.
        """
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            client=self._client,
        )


class ChunkedStream(tts.ChunkedStream):
    """
    Provides a streaming generator of TTS audio chunks.
    """

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
        client: openai.AsyncClient,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._client = client
        self._opts = opts

    async def _run(self):
        """
        Internal method that uses openai.AsyncClient's streaming interface
        to fetch TTS audio in chunks.
        """

        # Here we also set a single default for the streaming timeout
        oai_stream = self._client.audio.speech.with_streaming_response.create(
            input=self.input_text,
            model=self._opts.model,
            voice=self._opts.voice,
            response_format="pcm",  # or 'mp3', 'wav', 'opus', etc.
            speed=self._opts.speed,
            timeout=httpx.Timeout(30.0),  # Single default to avoid ValueError
        )

        request_id = utils.shortuuid()

        # This helps us break up raw audio data into frames
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=OPENAI_TTS_SAMPLE_RATE,
            num_channels=OPENAI_TTS_CHANNELS,
        )

        try:
            async with oai_stream as stream:
                async for data in stream.iter_bytes():
                    # 'data' is raw audio bytes (PCM or otherwise)
                    for frame in audio_bstream.write(data):
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                frame=frame,
                                request_id=request_id,
                            )
                        )
                # Flush any remaining data in the buffer
                for frame in audio_bstream.flush():
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            frame=frame,
                            request_id=request_id,
                        )
                    )

        except openai.APITimeoutError:
            raise APITimeoutError()
        except openai.APIStatusError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
            )
        except Exception as e:
            raise APIConnectionError() from e