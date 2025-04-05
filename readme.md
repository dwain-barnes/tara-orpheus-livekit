# Tara - Local Voice Assistant with Orpheus TTS and LiveKit

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)

## 🔊 Overview

Tara is a fully local, zero-cost voice assistant that combines the power of Orpheus TTS, LiveKit, and local LLMs to create incredibly natural and expressive speech. This project eliminates the need for cloud-based API services by integrating:

- **Orpheus TTS** for human-like speech with natural intonation and emotion
- **LiveKit** for real-time voice communication
- **Local Whisper** for accurate speech-to-text conversion
- **Ollama** for running local large language models

The result is a voice assistant with remarkably natural speech capabilities, including emotional expressions like laughs, sighs, and gasps - all running completely on your local machine.

## ✨ Features

- 🎯 **100% Local** - No API costs or cloud dependencies
- 🗣️ **Expressive Speech** - Natural intonation, rhythm, and emotional expressions
- 🎭 **Emotion Tags** - Simple text-based tags to control emotion and expression
- 🎙️ **Real-time Conversation** - Fluid interaction through LiveKit
- 🧠 **Local LLM Integration** - Uses Ollama to run powerful models locally
- 👂 **Advanced Speech Recognition** - Fast local transcription with Whisper

## 📋 Prerequisites

Before running Tara, you'll need:

- Python 3.9+
- [Orpheus TTS FastAPI Server](https://github.com/Lex-au/Orpheus-FastAPI) installed and running
- [Ollama](https://ollama.ai/) installed with the LLama3.2 model
- [Faster Whisper] or compatible STT system

## 🚀 Installation

1. First, set up the Orpheus TTS server:
   ```bash
   # Clone and set up the Orpheus FastAPI server
   git clone https://github.com/Lex-au/Orpheus-FastAPI
   cd Orpheus-FastAPI
   # Follow the setup instructions from the repository
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/dwain-barnes/tara-orpheus-livekit
   cd tara-orpheus-livekit
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env.local` file with your configuration:
   ```
   # Your LiveKit configuration
   LIVEKIT_URL=your_livekit_url
   LIVEKIT_API_KEY=your_api_key
   LIVEKIT_API_SECRET=your_api_secret
   ```

## 💬 Usage

1. Make sure the Orpheus TTS server is running (default: http://localhost:5005)
2. Make sure Ollama is running with the llama3.2 model loaded
3. Run the voice assistant:
   ```bash
   python tara.py
   ```

4. Connect to the LiveKit room and start interacting with Tara

## 🔧 How It Works

The system consists of several integrated components:

1. **Speech-to-Text (STT)**: Uses Faster Whisper for local transcription
2. **Language Model**: Connects to a local Ollama instance running LLama3.2
3. **Text-to-Speech (TTS)**: Modified OpenAI TTS module that connects to Orpheus TTS
4. **Voice Pipeline**: Handles the flow between components via LiveKit

Tara uses special text tags to express emotions in speech:
- `<giggle>`, `<laugh>`, `<chuckle>` for humor
- `<sigh>`, `<groan>` for showing disappointment or frustration
- `<gasp>`, `<cough>`, `<sniffle>`, `<yawn>` for other human-like expressions

## 🔄 Customization

You can modify the `tara.py` file to:
- Change the voice by editing the `voice` parameter in the TTS setup
- Modify the personality by editing the system prompt
- Adjust the LLM model by changing the Ollama model name
- Configure different endpoints for any of the services

## 📝 Code Explanation

The main workflow in `tara.py`:

```python
# 1) Speech-to-Text with FasterWhisper
stt_plugin = openai.STT.with_faster_whisper(model="Systran/faster-distil-whisper-large-v3")

# 2) Language Model from Ollama
llm_plugin = openai.LLM(
    base_url="http://localhost:11434/v1",
    api_key=os.environ.get("12343"),
    model="llama3.2:latest",
)

# 3) Text-to-Speech using Orpheus TTS
tts_plugin = TTS.create_orpheus_client(
    voice="tara",
    base_url="http://localhost:5005/v1" 
)

# 4) Create a VoicePipelineAgent
agent = VoicePipelineAgent(
    vad=ctx.proc.userdata["vad"],
    stt=stt_plugin,
    llm=llm_plugin,
    tts=tts_plugin,
    chat_ctx=initial_ctx,
    turn_detector=turn_detector.EOUModel(),
)
```



## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) for the incredible speech synthesis
- [LiveKit](https://livekit.io/) for the real-time communication platform
- [Lex-au](https://github.com/Lex-au/Orpheus-FastAPI) for the Orpheus FastAPI implementation
