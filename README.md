# Pipecat Flow Voicemail Bot

A voice bot that can detect voicemail systems vs. humans and respond appropriately using Pipecat AI.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Daily.co API key
- Cartesia API key  
- Google API key

### Setup

1. **Clone and navigate to the project directory:**
   ```bash
   cd pipecat-flow-voicemail
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root with your API keys:
   ```
   DAILY_API_KEY=your_daily_api_key_here
   CARTESIA_API_KEY=your_cartesia_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   TO_PHONE_NUMBER=+1234567890
   PARTNER_PHONE_NUMBER=+1987654321
   ```

### Running the Bot

Run the bot with:
```bash
python bot.py
```

The bot will:
1. Create a Daily.co room
2. Call the specified phone number
3. Detect if it reaches a voicemail system or human
4. Respond appropriately (leave voicemail message or have conversation)

### Deactivating the Virtual Environment

When you're done, deactivate the virtual environment:
```bash
deactivate
```

## Features

- **Voicemail Detection**: Automatically detects voicemail systems vs. humans
- **Dual Response Modes**: Leaves voicemail messages or engages in conversation
- **Flow-based Conversation**: Uses Pipecat Flows for structured human interactions
- **Multi-phase Pipeline**: Separate pipelines for detection and conversation phases