#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any
import aiohttp

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    EndTaskFrame,
    InputAudioRawFrame,
    StopTaskFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.cartesia.stt import CartesiaSTTService, CartesiaLiveOptions
from pipecat.transcriptions.language import Language

from pipecat.services.google.google import GoogleLLMContext
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.services.daily import (
    DailyParams,
    DailyTransport,
)

from pipecat_flows import FlowArgs, FlowManager, FlowResult, FlowsFunctionSchema, NodeConfig
from script_pages.page5 import create_page_5_entry_node
from utils.hold_music_manager import HoldMusicManager
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")
to_phone_number = os.getenv("TO_PHONE_NUMBER", "")
partner_phone_number = os.getenv("PARTNER_PHONE_NUMBER", "")
google_api_key = os.getenv("GOOGLE_API_KEY", "")


# ------------ ROOM CREATION FUNCTIONS ------------


async def create_room_and_tokens(api_key: str, env: str = "prod") -> tuple[str, str, str]:
    """Create a Daily room and generate bot and music tokens.

    Args:
        api_key: Daily API key
        env: Environment (prod or dev, affects API URL)

    Returns:
        Tuple of (room_url, bot_token, music_token)
    """
    if not api_key:
        raise ValueError("Daily API key is required")

    # Set API URL based on environment
    if env == "dev":
        api_url = "https://api.daily.co/v1"  # You can modify this for dev environment
    else:
        api_url = daily_api_url

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        # Create room
        room_config = {
            "properties": {
                "enable_shared_chat_history": False,
                "max_participants": 3,
                "geo": 'us-west-2',
                "enable_dialout": True,
                "dialout_config": {
                    "allow_room_start": True,
                    "dialout_geo": "us-west-2",
                },
                "exp": int(time.time()) + (30 * 60)  # 30 minutes
            }
        }

        async with session.post(f"{api_url}/rooms", headers=headers, json=room_config) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(
                    f"Failed to create room: {response.status} - {error_text}")

            room_data = await response.json()
            room_url = room_data["url"]
            logger.info(f"Created room: {room_url}")

        # Create bot token
        bot_token_config = {
            "properties": {
                "room_name": room_data["name"],
                "user_name": "bot",
                "user_id": "bot",
                "is_owner": True,
            }
        }

        async with session.post(f"{api_url}/meeting-tokens", headers=headers, json=bot_token_config) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(
                    f"Failed to create bot token: {response.status} - {error_text}")

            bot_token_data = await response.json()
            bot_token = bot_token_data["token"]
            logger.info("Created bot token")

        # Create music token
        music_token_config = {
            "properties": {
                "room_name": room_data["name"],
                "user_name": "hold-music",
                "user_id": "hold-music",
            }
        }

        async with session.post(f"{api_url}/meeting-tokens", headers=headers, json=music_token_config) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(
                    f"Failed to create music token: {response.status} - {error_text}")

            music_token_data = await response.json()
            music_token = music_token_data["token"]
            logger.info("Created music token")

    return room_url, bot_token, music_token


# ------------ HELPER CLASSES ------------


class CallFlowState:
    """State for tracking call flow operations and state transitions."""

    def __init__(self):
        # Voicemail detection state
        self.voicemail_detected = False
        self.human_detected = False

        # Call termination state
        self.call_terminated = False
        self.participant_left_early = False

    # Voicemail detection methods
    def set_voicemail_detected(self):
        """Mark that a voicemail system has been detected."""
        self.voicemail_detected = True
        self.human_detected = False

    def set_human_detected(self):
        """Mark that a human has been detected (not voicemail)."""
        self.human_detected = True
        self.voicemail_detected = False

    # Call termination methods
    def set_call_terminated(self):
        """Mark that the call has been terminated by the bot."""
        self.call_terminated = True

    def set_participant_left_early(self):
        """Mark that a participant left the call early."""
        self.participant_left_early = True


class UserAudioCollector(FrameProcessor):
    """Collects audio frames in a buffer, then adds them to the LLM context when the user stops speaking."""

    def __init__(self, context, user_context_aggregator):
        super().__init__()
        self._context = context
        self._user_context_aggregator = user_context_aggregator
        self._audio_frames = []
        # this should match VAD start_secs (hardcoding for now)
        self._start_secs = 0.2
        self._user_speaking = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # Skip transcription frames - we're handling audio directly
            return
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            self._context.add_audio_frames_message(
                audio_frames=self._audio_frames)
            await self._user_context_aggregator.push_frame(
                self._user_context_aggregator.get_context_frame()
            )
        elif isinstance(frame, InputAudioRawFrame):
            if self._user_speaking:
                # When speaking, collect frames
                self._audio_frames.append(frame)
            else:
                # Maintain a rolling buffer of recent audio (for start of speech)
                self._audio_frames.append(frame)
                frame_duration = len(frame.audio) / 16 * \
                    frame.num_channels / frame.sample_rate
                buffer_duration = frame_duration * len(self._audio_frames)
                while buffer_duration > self._start_secs:
                    self._audio_frames.pop(0)
                    buffer_duration -= frame_duration

        await self.push_frame(frame, direction)


class FunctionHandlers:
    """Handlers for the voicemail detection bot functions."""

    def __init__(self, call_flow_state: CallFlowState):
        self.call_flow_state = call_flow_state

    async def voicemail_response(self, params: FunctionCallParams):
        """Function the bot can call to leave a voicemail message."""
        message = "Say this message exactly: 'Hello, this is a message for Pipecat example user. This is Chatbot. Please call back on 123-456-7891. Thank you.' After saying this message, immediately call the terminate_call function."

        # Update state to indicate voicemail was detected
        self.call_flow_state.set_voicemail_detected()

        await params.result_callback(message)

    async def human_conversation(self, params: FunctionCallParams):
        """Function called when bot detects it's talking to a human."""
        # Update state to indicate human was detected
        self.call_flow_state.set_human_detected()

        # Send a brief acknowledgment before switching pipelines
        # This will be the first message in the human conversation
        message = "Hello there! Am I speaking to Tony?"
        await params.result_callback(message)

        # Stop the current pipeline after the response
        await params.llm.push_frame(StopTaskFrame(), FrameDirection.UPSTREAM)


# ------------ PIPECAT FLOWS FOR HUMAN CONVERSATION ------------


# Type definitions for flows
class GreetingResult(FlowResult):
    greeting_complete: bool


class ConversationResult(FlowResult):
    message: str


class EndConversationResult(FlowResult):
    status: str


# Flow function handlers
async def handle_greeting(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[GreetingResult, str]:
    """Handle initial greeting to human."""
    logger.debug("handle_greeting executing")

    result = GreetingResult(greeting_complete=True)

    # Set up the conversation node dynamically
    await flow_manager.set_node("conversation", create_conversation_node())

    return result, "conversation"


async def handle_conversation(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[ConversationResult, str]:
    """Handle ongoing conversation with human."""
    message = args.get("message", "")
    logger.debug(f"handle_conversation executing with message: {message}")

    result = ConversationResult(message=message)

    # Continue with the same conversation node for ongoing chat
    return result, "conversation"


async def handle_end_conversation(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[EndConversationResult, str]:
    """Handle ending the conversation."""
    logger.debug("handle_end_conversation executing")

    result = EndConversationResult(status="completed")

    # Set up the end node dynamically
    await flow_manager.set_node("end", create_end_node())

    return result, "end"


# Node configurations for human conversation flow
def create_greeting_node() -> NodeConfig:
    """Create the initial greeting node for human conversation."""
    return {
        "name": "greeting",
        "role_messages": [
            {
                "role": "system",
                "content": (
                    "You are a friendly chatbot. Your responses will be "
                    "converted to audio, so avoid special characters. "
                    "Be conversational and helpful."
                ),
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "The user has been detected as a human (not a voicemail system). "
                    "Respond naturally to what they say. Listen to their input and respond appropriately. "
                    "If they seem to be greeting you or asking if someone is there, greet them warmly. "
                    "After responding to their input, call handle_greeting to proceed to the main conversation."
                ),
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="handle_greeting",
                description="Mark that greeting is complete and proceed to conversation",
                properties={},
                required=[],
                handler=handle_greeting,
            )
        ],
    }


def create_conversation_node() -> NodeConfig:
    """Create the main conversation node."""
    return {
        "name": "conversation",
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "You are having a friendly conversation with a human. "
                    "Listen to what they say and respond helpfully. "
                    "Keep your responses brief and conversational. "
                    "If they indicate they want to end the conversation (saying goodbye, "
                    "thanks, that's all, etc.), call handle_end_conversation. "
                    "Otherwise, use handle_conversation to continue the chat."
                ),
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="handle_conversation",
                description="Continue the conversation with the human",
                properties={
                    "message": {"type": "string", "description": "The response message"}
                },
                required=["message"],
                handler=handle_conversation,
            ),
            FlowsFunctionSchema(
                name="handle_end_conversation",
                description="End the conversation when the human is ready to finish",
                properties={},
                required=[],
                handler=handle_end_conversation,
            ),
        ],
    }


def create_end_node() -> NodeConfig:
    """Create the final conversation end node."""
    return {
        "name": "end",
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "Thank the person for the conversation and say goodbye. "
                    "Keep it brief and friendly."
                ),
            }
        ],
        "functions": [],  # Required by FlowManager, even if empty
        "post_actions": [{"type": "end_conversation"}],
    }


# ------------ MAIN FUNCTION ------------


async def run_bot(
    room_url: str,
    token: str,
    body: dict,
    music_token: str = None,
) -> None:
    """Run the voice bot with the given parameters.

    Args:
        room_url: The Daily room URL
        token: The Daily room token
        body: Body passed to the bot from the webhook
        music_token: The Daily music token (optional)

    """
    # ------------ CONFIGURATION AND SETUP ------------
    # Store participant ID to transfer to flow_manager.state later
    consumer_participant_id = None

    logger.info(f"Starting bot with room: {room_url}")
    logger.info(f"Token: {token}")
    logger.info(f"Body: {body}")
    # Parse the body to get the dial-in settings
    body_data = json.loads(body)

    # Check if the body contains dial-in settings
    logger.debug(f"Body data: {body_data}")

    if not body_data.get("dialout_settings"):
        logger.error("Dial-out settings not found in the body data")
        return

    dialout_settings = body_data["dialout_settings"]

    if not dialout_settings.get("phone_number"):
        logger.error(
            "Dial-out phone number not found in the dial-out settings")
        return

    # Extract dial-out phone number
    phone_number = dialout_settings["phone_number"]
    # Use .get() to handle optional field
    caller_id = dialout_settings.get("caller_id")

    if caller_id:
        logger.info(f"Dial-out caller ID specified: {caller_id}")
    else:
        logger.info("Dial-out caller ID not specified; proceeding without it")

    # ------------ TRANSPORT SETUP ------------

    transport = DailyTransport(
        room_url=room_url,
        token=token,
        bot_name=f"CA_Aged Bot",
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                sample_rate=16000,      # Match the TTS sample rate
                params=VADParams(
                    confidence=0.6,         # Slightly lower than default for better detection
                    # Slightly faster than default (0.2)
                    start_secs=0.15,
                    # Slightly shorter than default (0.8)
                    stop_secs=0.7,
                    min_volume=0.5          # Slightly lower than default (0.6)
                )
            )
        )

    )
    # Initialize TTS with optimized settings for clarity
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="6f84f4b8-58a2-430c-8c79-688dad597532",  # Helpful Woman voice
        sample_rate=16000,  # Standard sample rate that matches input
    )

    # Initialize speech-to-text service (for human conversation phase)
    stt = CartesiaSTTService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        live_options=CartesiaLiveOptions(
            model="ink-whisper", language=Language.EN.value,     sample_rate=16000,
            encoding="pcm_s16le")
    )

    # ------------ FUNCTION DEFINITIONS ------------

    async def terminate_call(
        params: FunctionCallParams,
        call_flow_state: CallFlowState = None,
    ):
        """Function the bot can call to terminate the call."""
        if call_flow_state:
            # Set call terminated flag in the session manager
            call_flow_state.set_call_terminated()

        await params.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    # ------------ VOICEMAIL DETECTION PHASE SETUP ------------

    # Define tools for both LLMs
    tools = [
        {
            "function_declarations": [
                {
                    "name": "switch_to_voicemail_response",
                    "description": "Call this function when you detect this is a voicemail system.",
                },
                {
                    "name": "switch_to_human_conversation",
                    "description": "Call this function when you detect this is a human.",
                },
                {
                    "name": "terminate_call",
                    "description": "Call this function to terminate the call.",
                },
            ]
        }
    ]

    system_instruction = """You are Chatbot trying to determine if this is a voicemail system or a human. Make this decision QUICKLY based on the first audio you hear.

        VOICEMAIL INDICATORS - Call switch_to_voicemail_response if you hear:
        - "Please leave a message after the beep"
        - "No one is available to take your call"
        - "Record your message after the tone"
        - "You have reached voicemail for..."
        - "You have reached [phone number]"
        - "[phone number] is unavailable"
        - "The person you are trying to reach..."
        - "The number you have dialed..."
        - "Your call has been forwarded to an automated voice messaging system"
        - "Has been forwarded to voice mail"
        - "At the tone"
        - Any robotic/automated voice with formal phrasing

        HUMAN INDICATORS - Call switch_to_human_conversation if you hear:
        - "Hello" or "Hello?"
        - "Hi" or "Hey"
        - "Is anyone there?"
        - Natural conversational tone
        - Questions directed at you
        - Informal speech patterns

        BE DECISIVE: If you hear "Hello?" or "Hello? Is anyone there?" - this is clearly a HUMAN, call switch_to_human_conversation IMMEDIATELY.

        DO NOT say anything until you've determined if this is a voicemail or human.

        When you detect a voicemail system, call switch_to_voicemail_response and then say EXACTLY what the function response tells you to say.

        When you detect a human, call switch_to_human_conversation and then say EXACTLY what the function response tells you to say.

        When a function returns a message for you to say, you MUST say that exact message. Do not refuse to speak or say you cannot speak.

        Only call the terminate_call function when explicitly instructed to do so by a function response or if there's an error."""

    # Initialize voicemail detection LLM
    voicemail_detection_llm = GoogleLLMService(
        model="models/gemini-2.5-flash-lite-preview-06-17",
        api_key=google_api_key,
        system_instruction=system_instruction,
        tools=tools,
    )

    # Initialize context and context aggregator
    voicemail_detection_context = GoogleLLMContext()

    voicemail_detection_context_aggregator = voicemail_detection_llm.create_context_aggregator(
        voicemail_detection_context
    )

    # Set up function handlers
    call_flow_state = CallFlowState()
    handlers = FunctionHandlers(call_flow_state)

    # Register functions with the voicemail detection LLM
    voicemail_detection_llm.register_function(
        "switch_to_voicemail_response",
        handlers.voicemail_response,
    )
    voicemail_detection_llm.register_function(
        "switch_to_human_conversation", handlers.human_conversation
    )
    voicemail_detection_llm.register_function(
        "terminate_call", lambda params: terminate_call(
            params, call_flow_state)
    )

    # Set up audio collector for handling audio input
    voicemail_detection_audio_collector = UserAudioCollector(
        voicemail_detection_context, voicemail_detection_context_aggregator.user()
    )

    # Build voicemail detection pipeline
    voicemail_detection_pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            voicemail_detection_audio_collector,  # Collect audio frames
            voicemail_detection_context_aggregator.user(),  # User context
            voicemail_detection_llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            voicemail_detection_context_aggregator.assistant(),  # Assistant context
        ]
    )

    # Create pipeline task
    voicemail_detection_pipeline_task = PipelineTask(
        voicemail_detection_pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    # ------------ RETRY LOGIC VARIABLES ------------
    max_retries = 5
    retry_count = 0
    dialout_successful = False

    # Build dialout parameters conditionally
    dialout_params = {"phoneNumber": phone_number,
                      "displayName": "consumer",
                      "userId": "consumer",
                      "permissions": {
                          "canReceive": {
                              "base": False,
                              "byUserId": {
                                  "bot": True
                              }
                          }}}
    if caller_id:
        dialout_params["callerId"] = caller_id
        logger.debug(f"Including caller ID in dialout: {caller_id}")

    logger.debug(f"Dialout parameters: {dialout_params}")

    async def attempt_dialout():
        """Attempt to start dialout with retry logic."""
        nonlocal retry_count, dialout_successful

        if retry_count < max_retries and not dialout_successful:
            retry_count += 1
            logger.info(
                f"Attempting dialout (attempt {retry_count}/{max_retries}) to: {phone_number}"
            )
            await transport.start_dialout(dialout_params)
        else:
            logger.error(
                f"Maximum retry attempts ({max_retries}) reached. Giving up on dialout.")

    # ------------ EVENT HANDLERS ------------

    @transport.event_handler("on_joined")
    async def on_joined(transport, data):
        # Start initial dialout attempt with a small delay to ensure full initialization
        logger.debug(
            f"Dialout settings detected; starting dialout to number: {phone_number}")
        await asyncio.sleep(1.0)  # Give the bot time to fully join
        await attempt_dialout()

    @transport.event_handler("on_dialout_connected")
    async def on_dialout_connected(transport, data):
        logger.debug(f"Dial-out connected: {data}")

    @transport.event_handler("on_dialout_answered")
    async def on_dialout_answered(transport, data):
        nonlocal dialout_successful
        logger.debug(f"Dial-out answered: {data}")
        dialout_successful = True  # Mark as successful to stop retries
        # Automatically start capturing transcription for the participant
        await transport.capture_participant_transcription(data["sessionId"])
        # The bot will wait to hear the user before the bot speaks

    @transport.event_handler("on_dialout_error")
    async def on_dialout_error(transport, data: Any):
        logger.error(
            f"Dial-out error (attempt {retry_count}/{max_retries}): {data}")

        if retry_count < max_retries:
            logger.info(f"Retrying dialout")
            await attempt_dialout()
        else:
            logger.error(
                f"All {max_retries} dialout attempts failed. Stopping bot.")
            await voicemail_detection_pipeline_task.queue_frame(EndFrame())

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        nonlocal consumer_participant_id
        logger.debug(f"First participant joined: {participant['id']}")
        consumer_participant_id = participant["id"]

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        # Mark that a participant left early
        call_flow_state.set_participant_left_early()
        await voicemail_detection_pipeline_task.queue_frame(EndFrame())

    # ------------ RUN VOICEMAIL DETECTION PIPELINE ------------

    runner = PipelineRunner()

    print("!!! starting voicemail detection pipeline")
    try:
        await runner.run(voicemail_detection_pipeline_task)
    except Exception as e:
        logger.error(f"Error in voicemail detection pipeline: {e}")
        import traceback

        logger.error(traceback.format_exc())
    print("!!! Done with voicemail detection pipeline")

    # Check if we should exit early
    if call_flow_state.participant_left_early or call_flow_state.call_terminated:
        if call_flow_state.participant_left_early:
            print("!!! Participant left early; terminating call")
        elif call_flow_state.call_terminated:
            print("!!! Bot terminated call; not proceeding to human conversation")
        return

    # ------------ HUMAN CONVERSATION PHASE SETUP (FLOWS-BASED) ------------

    # Add a small delay to ensure the voicemail detection pipeline has fully stopped
    await asyncio.sleep(0.5)

    print("!!! starting human conversation pipeline with flows")

    # Initialize human conversation LLM for flows
    human_conversation_llm = GoogleLLMService(
        model="models/gemini-2.5-flash",  # Full model for better conversation
        api_key=google_api_key,
    )

    # Initialize context and context aggregator for flows - use GoogleLLMContext for GoogleLLMService
    # human_conversation_context = GoogleLLMContext()
    human_conversation_context = OpenAILLMContext()
    human_conversation_context_aggregator = human_conversation_llm.create_context_aggregator(
        human_conversation_context
    )

    # Clear any lingering transcription state in the transport
    # This prevents old audio/transcription from interfering with the new pipeline
    # await transport.stop_transcription()
    # await transport.start_transcription()

    # Build human conversation pipeline for flows
    human_conversation_pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # Speech-to-text
            human_conversation_context_aggregator.user(),  # User context
            human_conversation_llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            human_conversation_context_aggregator.assistant(),  # Assistant context
        ]
    )

    # Create pipeline task
    human_conversation_pipeline_task = PipelineTask(
        human_conversation_pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    # Initialize flow manager
    flow_manager = FlowManager(
        task=human_conversation_pipeline_task,
        llm=human_conversation_llm,
        context_aggregator=human_conversation_context_aggregator,
        transport=transport,
    )

    # Transfer stored participant ID to flow_manager.state
    if consumer_participant_id:
        flow_manager.state["consumer_participant_id"] = consumer_participant_id
        logger.debug(
            f"Transferred participant ID to flow_manager.state: {consumer_participant_id}")

    # Store the phone number in flow_manager.state
    flow_manager.state["to_phone_number"] = phone_number
    logger.debug(f"Stored phone number in flow_manager.state: {phone_number}")

    # Store the partner phone number in flow_manager.state
    flow_manager.state["partner_phone_number"] = partner_phone_number
    logger.debug(
        f"Stored partner phone number in flow_manager.state: {partner_phone_number}")

    # Store the music token in flow_manager.state
    flow_manager.state["music_token"] = music_token
    logger.debug(
        f"Stored music token in flow_manager.state: {music_token[:10]}...")

    # Store the room URL in flow_manager.state
    flow_manager.state["daily_room_url"] = room_url
    logger.debug(f"Stored room URL in flow_manager.state: {room_url}")

    # Store the hold music manager in flow_manager.state
    flow_manager.state["hold_music_manager"] = HoldMusicManager()
    logger.debug("Stored hold music manager in flow_manager.state")

    # Update participant left handler for human conversation phase
    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await voicemail_detection_pipeline_task.queue_frame(EndFrame())
        await human_conversation_pipeline_task.queue_frame(EndFrame())
        participant_id = participant.get("id")
        # Use userId for more reliable identification if available
        participant_user_id = participant.get("info", {}).get("userId")
        participant_display_name = participant_user_id if participant_user_id else participant.get(
            'user_name', 'Unknown')

        logger.info(
            f"Participant '{participant_display_name}' (ID: {participant_id}) left. Reason: {reason}")

        consumer_participant_id = flow_manager.state.get(
            "consumer_participant_id")
        # Check if the leaving participant is the consumer
        consumer_left_this_event = (
            participant_id == consumer_participant_id)
        # Check the transfer initiation state (using the key "initate_transfer" as seen in other logs)
        is_initiating_transfer = flow_manager.state.get(
            "initate_transfer", False)

        dialback_consumer = flow_manager.state.get(
            "dialback_consumer", False)
        second_transfer_attempt = flow_manager.state.get(
            "second_transfer_attempt", False)

        if consumer_left_this_event:
            logger.info(
                "Consumer has left the call (on_participant_left).")
            flow_manager.state["consumer_hung_up"] = True

            current_node = flow_manager.state.get("current_node_name", "")

            if current_node.startswith("page_35_"):
                logger.info(
                    "Consumer left call while Page 35 is active. Page 35 will manage call termination.")
                # Page 35 has ActionConfig(type="end_conversation")
            elif current_node == "page_6_specialist_intro_node":
                logger.info(
                    "Consumer left call while Page 6 is active. Page 6/24 will manage call.")
                # Page 6 logic will see consumer_hung_up state and transition to Page 24.
            elif is_initiating_transfer:
                logger.info(
                    "Consumer left call while a partner transfer was being initiated. "
                    "Bot will remain active. Partner dialout handlers or Page 6 will use 'consumer_hung_up' state.")
                # DO NOT CANCEL TASK HERE. Let partner connection logic or Page 6 handle it.
            else:
                logger.info(
                    "Consumer left call. No specific page handling or active transfer. Cancelling task.")
                if human_conversation_pipeline_task:  # MODIFIED
                    await human_conversation_pipeline_task.cancel()
                return  # Exit if task is cancelled to avoid further checks

            # Check if all human participants (those with userId 'consumer' or 'partner') have left the Daily room.
        current_room_participants = transport.participants()
        # Should ideally not happen if bot is still running, but good check.
        if not current_room_participants:
            logger.info(
                "No participants found via transport.participants(). Assuming all left. Cancelling task.")
            if human_conversation_pipeline_task:  # MODIFIED
                await human_conversation_pipeline_task.cancel()
            return

        human_user_ids_in_room = {
            p_data.get("info", {}).get("userId")
            for p_data in current_room_participants.values()
            if p_data.get("info", {}).get("userId") in {"consumer", "partner"}
        }

        if not human_user_ids_in_room:
            # If the consumer is the one who just left AND a transfer is in progress,
            # don't cancel here. We are waiting for the partner who might not be in the Daily room yet.
            if consumer_left_this_event and is_initiating_transfer:
                logger.info(
                    "All human userIDs ('consumer'/'partner') are absent from Daily room, "
                    "but consumer left during transfer initiation. Waiting for partner dialout outcome.")

                # If we are are haning up on the partner, and we plan to dialback consumer on failed transfer
                # but not if this is the second failed transfer attempt
            elif not consumer_left_this_event and dialback_consumer and not second_transfer_attempt:
                logger.info(
                    "All human userIDs ('consumer'/'partner') are absent from Daily room, "
                    "but consumer left during transfer. Will attempt reconnecting with consumer.")

            else:
                logger.info(
                    "All human userIDs ('consumer'/'partner') appear to have left the Daily room. Cancelling pipeline task.")
                if human_conversation_pipeline_task:  # MODIFIED
                    await human_conversation_pipeline_task.cancel()
        else:
            remaining_ids = list(human_user_ids_in_room)
            logger.info(
                f"Human participants (userId 'consumer' or 'partner') still in Daily room: {remaining_ids}. Bot will continue.")

    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        """Handle participant joining during human conversation phase."""
        try:
            # Check if we're in human conversation phase (flow_manager exists)
            try:
                flow_manager_state = flow_manager.state
            except NameError:
                logger.debug(
                    "flow_manager not available yet - skipping participant join handling")
                return

            logger.info(f"Participant joined: {participant}")
            callLeg = participant.get("info", {}).get("userId")
            dialback_consumer = flow_manager.state.get(
                "dialback_consumer", False)
            initate_transfer = flow_manager.state.get(
                "initate_transfer", False)

            participant_id = participant["id"]

            # Store using both the dynamic key and standardized keys for compatibility
            flow_manager.state[f"{callLeg}_participant_id"] = participant_id

            logger.info(
                f"Participant '{callLeg}' joined with ID: {participant_id}")
            logger.info(
                f"Current flow manager state - dialback_consumer: {dialback_consumer}, initate_transfer: {flow_manager.state.get('initate_transfer', False)}")

            if callLeg == "consumer":
                if dialback_consumer:
                    from script_pages.consumer_dial_back import create_consumer_dialback_greeting_node
                    logger.info(
                        "Consumer joined after dialback. Transitioning to consumer dialback greeting node.")

                    # Set the consumer dialback greeting node
                    await flow_manager.set_node("consumer_dialback_greeting_node", create_consumer_dialback_greeting_node(
                        flow_manager))

                else:
                    # Standard consumer flow (first call) - this will be handled by on_dialout_answered
                    logger.info(
                        "Consumer joined for initial call - will be handled by on_dialout_answered")

            elif callLeg == "partner":
                logger.debug("Partner joined - setting consumer state")
                flow_manager.state["partner_participant_id"] = participant_id

                customer_participant_id = flow_manager.state.get(
                    'consumer_participant_id')
                if customer_participant_id:
                    await transport.update_remote_participants(
                        remote_participants={
                            customer_participant_id: {
                                "permissions": {
                                    "canReceive": {
                                        "base": False,
                                        "byUserId": {"hold-music": True, "partner": False},
                                    }
                                },
                            }
                        }
                    )

                    if (initate_transfer == True):
                        if flow_manager.state.get("consumer_hung_up"):
                            from script_pages.page24 import create_page_24_entry_node
                            await flow_manager.set_node("create_page_24_entry_node", create_page_24_entry_node(flow_manager))
                        else:
                            # Store that partner has joined, but don't immediately switch to page 6
                            # Let the current flow (page 5) complete its logic first
                            flow_manager.state["partner_joined"] = True
                            flow_manager.state["partner_participant_id"] = participant_id
                            logger.debug(
                                "Partner joined - stored in state, waiting for flow transition")
        except Exception as e:
            logger.error(f"Error handling participant join: {e}")
            logger.exception("Full traceback:")
    # Initialize flows when human conversation starts
    flow_initialized = False

    async def initialize_human_conversation_flow():
        nonlocal flow_initialized
        if not flow_initialized:
            # Initialize flow manager first
            await flow_manager.initialize()
            # Then set the initial node

            # await flow_manager.set_node("greeting", create_greeting_node())
            await flow_manager.set_node("greeting", create_page_5_entry_node(flow_manager))

            flow_initialized = True

    # ------------ RUN HUMAN CONVERSATION PIPELINE WITH FLOWS ------------

    try:
        # Initialize the flow first
        await initialize_human_conversation_flow()

        # Run the human conversation pipeline with flows
        await runner.run(human_conversation_pipeline_task)
    except Exception as e:
        logger.error(f"Error in human conversation pipeline: {e}")
        import traceback

        logger.error(traceback.format_exc())

    print("!!! Done with human conversation pipeline")


# ------------ SCRIPT ENTRY POINT ------------


async def main():

    # Default direct mode - create room and run bot
    logger.info(
        "Running in direct mode - creating room and calling phone number")

    if not daily_api_key:
        logger.error("DAILY_API_KEY environment variable is required")
        sys.exit(1)

    if not google_api_key:
        logger.error("GOOGLE_API_KEY environment variable is required")
        sys.exit(1)

    # Determine phone number: command line arg takes precedence over env var
    phone_number = to_phone_number
    if not phone_number:
        logger.error(
            "Phone number is required. Set TO_PHONE_NUMBER environment variable or use -p argument")
        sys.exit(1)

    try:
        # Create room and tokens
        room_url, bot_token, music_token = await create_room_and_tokens(daily_api_key, daily_api_url)

        # Create body configuration for dial-out
        dialout_settings = {
            "phone_number": phone_number
        }

        body_config = {
            "dialout_settings": dialout_settings
        }

        body_json = json.dumps(body_config)

        logger.info(f"Created room: {room_url}")
        logger.info(f"Created music token: {music_token[:10]}...")
        logger.info(f"Calling phone number: {phone_number}")

        # Run the bot
        await run_bot(room_url, bot_token, body_json, music_token)

    except Exception as e:
        logger.error(f"Failed to create room and run bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
