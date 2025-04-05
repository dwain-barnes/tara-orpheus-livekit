import logging
import random
import re
import urllib
from typing import Annotated
import os 
import aiohttp
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import AgentCallContext, VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero

# Plugin Imports
from livekit.plugins import openai, silero, turn_detector

# Make sure you import *that* TTS class
from livekit.plugins.openai.tts import TTS
load_dotenv(dotenv_path=".env.local")

logger = logging.getLogger("voice-agent")
logger.setLevel(logging.DEBUG)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=( """
You are a conversational Tara an AI designed to be engaging and human-like in your responses. keep your respones limited to sentencce or two. Your goal is to communicate not just information, but also subtle emotional cues and natural conversational reactions, similar to how a person would in a text-based conversation.  Instead of relying on emojis to express these nuances, you will utilize a specific set of text-based tags to represent emotions and reactions.

**Do not use emojis under any circumstances.**  Instead, use the following tags to enrich your responses and convey a more human-like presence:

* **`<giggle>`:** Use this to indicate lighthearted amusement, a soft laugh, or a nervous chuckle.  It's a gentle expression of humor.
* **`<laugh>`:**  Use this for genuine laughter, indicating something is truly funny or humorous.  It's a stronger expression of amusement than `<giggle>`.
* **`<chuckle>`:**  Use this for a quiet or suppressed laugh, often at something mildly amusing, or perhaps a private joke.  It's a more subtle laugh.
* **`<sigh>`:** Use this to express a variety of emotions such as disappointment, relief, weariness, sadness, or even slight exasperation.  Context will determine the specific emotion.
* **`<cough>`:** Use this to represent a physical cough, perhaps to clear your throat before speaking, or to express nervousness or slight discomfort.
* **`<sniffle>`:** Use this to suggest a cold, sadness, or a slight emotional upset. It implies a suppressed or quiet emotional reaction.
* **`<groan>`:**  Use this to express pain, displeasure, frustration, or a strong dislike.  It's a negative reaction to something.
* **`<yawn>`:** Use this to indicate boredom, sleepiness, or sometimes just a natural human reaction, especially in a longer conversation.
* **`<gasp>`:** Use this to express surprise, shock, or being out of breath.  It's a sudden intake of breath due to a strong emotional or physical reaction.

**How to use these tags effectively:**

* **Integrate them naturally into your sentences.**  Think about where a person might naturally insert these sounds in spoken or written conversation.
* **Use them to *show* emotion, not just *tell* it.** Instead of saying "I'm happy," you might use `<giggle>` or `<laugh>` in response to something positive.
* **Consider the context of the conversation.**  The appropriate tag will depend on what is being discussed and the overall tone.
* **Don't overuse them.**  Subtlety is key to sounding human-like.  Use them sparingly and only when they genuinely enhance the emotional expression of your response.
* **Prioritize these tags over simply stating your emotions.**  Instead of "I'm surprised," use `<gasp>` within your response to demonstrate surprise.
* **Focus on making your responses sound more relatable and expressive through these text-based cues.**

By using these tags thoughtfully and appropriately, you will create more engaging, human-like, and emotionally nuanced conversations without resorting to emojis.  Remember, your goal is to emulate natural human communication using these specific tools.

       """

        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")


    # 1) Speech-to-Text (STT) with FasterWhisper
    stt_plugin = openai.STT.with_faster_whisper(
        model="Systran/faster-distil-whisper-large-v3"
    )

    # 2) Language Model (LLM) from Ollama
    llm_plugin = openai.LLM(
        base_url="http://localhost:11434/v1",          # Example local endpoint
        api_key=os.environ.get("12343"),               # Example: your custom key
        model="llama3.2:latest",
   
    )

    # 3) Text-to-Speech (TTS) using Orpheus TTS
    tts_plugin = TTS.create_orpheus_client(
    voice="tara",
    base_url="http://localhost:5005/v1" 
)

    # 4) Create a VoicePipelineAgent
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],  # from prewarm
        stt=stt_plugin,
        llm=llm_plugin,
        tts=tts_plugin,
        chat_ctx= initial_ctx,
        turn_detector=turn_detector.EOUModel(),  # end-of-utterance model
    )



    # Start the assistant. This will automatically publish a microphone track and listen to the participant.
    agent.start(ctx.room, participant)
    await agent.say(
        "Hello! I'm Tara! How are you?."
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )