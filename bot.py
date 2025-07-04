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

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat_flows import FlowArgs, FlowManager, FlowResult, FlowsFunctionSchema, NodeConfig

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    EndTaskFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
    SystemFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMContext, GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.sync.event_notifier import EventNotifier
from pipecat.transports.services.daily import (
    DailyParams,
    DailyTransport,
)
from pipecatcloud import DailySessionArguments


load_dotenv(override=True)

logger.remove()
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")

use_prebuilt = False

# aiohttp session will be created in async context


# Simple constants for model states
VOICEMAIL_MODE = "voicemail"
HUMAN_MODE = "human"
MUTE_MODE = "mute"

VOICEMAIL_CONFIDENCE_THRESHOLD = 0.6
HUMAN_CONFIDENCE_THRESHOLD = 0.6

# ------------ FLOW MANAGER SETUP ------------


# ------------ PIPECAT FLOWS FOR HUMAN CONVERSATION ------------


# Type definitions for flows
class GreetingResult(FlowResult):
    greeting_complete: bool


class ConversationResult(FlowResult):
    message: str


class EndConversationResult(FlowResult):
    status: str


# Flow function handlers
async def handle_greeting(args: FlowArgs, flow_manager: FlowManager) -> tuple[GreetingResult, str]:
    """Handle initial greeting to human."""
    logger.debug("handle_greeting executing")

    result = GreetingResult(greeting_complete=True)

    # Get lead first name from flow manager state
    lead_first_name = flow_manager.state.get("lead_first_name")

    # Set up the conversation node dynamically with lead first name
    await flow_manager.set_node("conversation", create_conversation_node(lead_first_name))

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
                    """You are a friendly chatbot. Your responses will be
                    converted to audio, so avoid special characters.
                    Be conversational and helpful."""
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
                properties={"message": {"type": "string",
                                        "description": "The response message"}},
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


# ------------ SIMPLIFIED CLASSES ------------


class VoicemailDetectionObserver(BaseObserver):
    """Observes voicemail speaking patterns to know when voicemail is done."""

    def __init__(self, timeout: float = 5.0):
        super().__init__()
        self._processed_frames = set()
        self._timeout = timeout
        self._last_turn_time = 0
        self._voicemail_speaking = False

    async def on_push_frame(self, data: FramePushed):
        if data.frame.id in self._processed_frames:
            return
        self._processed_frames.add(data.frame.id)

        if isinstance(data.frame, UserStartedSpeakingFrame):
            self._voicemail_speaking = True
            self._last_turn_time = 0
        elif isinstance(data.frame, UserStoppedSpeakingFrame):
            self._last_turn_time = time.time()

    async def wait_for_voicemail(self):
        """Wait for voicemail to finish speaking."""
        while self._voicemail_speaking:
            logger.debug("📩️ Waiting for voicemail to finish")
            if self._last_turn_time:
                diff_time = time.time() - self._last_turn_time
                self._voicemail_speaking = diff_time < self._timeout
            if self._voicemail_speaking:
                await asyncio.sleep(0.5)


class OutputGate(FrameProcessor):
    """Simple gate that opens when notified."""

    def __init__(self, notifier, start_open: bool = False):
        super().__init__()
        self._gate_open = start_open
        self._frames_buffer = []
        self._notifier = notifier
        self._gate_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Always pass system frames and function call frames
        if isinstance(
            frame, (SystemFrame, EndFrame, FunctionCallInProgressFrame,
                    FunctionCallResultFrame)
        ):
            if isinstance(frame, StartFrame):
                await self._start()
            elif isinstance(frame, (CancelFrame, EndFrame)):
                await self._stop()
            elif isinstance(frame, StartInterruptionFrame):
                self._frames_buffer = []
                self._gate_open = False
            await self.push_frame(frame, direction)
            return

        # Only gate downstream frames
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if self._gate_open:
            await self.push_frame(frame, direction)
        else:
            # Buffer frames until gate opens
            self._frames_buffer.append((frame, direction))

    async def _start(self):
        self._frames_buffer = []
        if not self._gate_task:
            self._gate_task = self.create_task(self._gate_task_handler())

    async def _stop(self):
        if self._gate_task:
            await self.cancel_task(self._gate_task)
            self._gate_task = None

    async def _gate_task_handler(self):
        """Wait for notification to open gate."""
        while True:
            try:
                await self._notifier.wait()
                self._gate_open = True
                # Flush buffered frames
                for frame, direction in self._frames_buffer:
                    await self.push_frame(frame, direction)
                self._frames_buffer = []
                break  # Gate stays open
            except asyncio.CancelledError:
                break


class UserAudioCollector(FrameProcessor):
    """Collects audio frames for the LLM context."""

    def __init__(self, context, user_context_aggregator):
        super().__init__()
        self._context = context
        self._user_context_aggregator = user_context_aggregator
        self._audio_frames = []
        self._start_secs = 0.2
        self._user_speaking = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
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
                self._audio_frames.append(frame)
            else:
                # Maintain rolling buffer
                self._audio_frames.append(frame)
                frame_duration = len(frame.audio) / 16 * \
                    frame.num_channels / frame.sample_rate
                buffer_duration = frame_duration * len(self._audio_frames)
                while buffer_duration > self._start_secs:
                    self._audio_frames.pop(0)
                    buffer_duration -= frame_duration

        await self.push_frame(frame, direction)

# ------------ MAIN FUNCTION ------------


async def run_bot(room_url: str, token: str, body: dict) -> None:
    """Run the voice bot with parallel pipeline architecture."""

    # ------------ SETUP ------------
    logger.info(f"Starting bot with room: {room_url}")

    # Create aiohttp session for API calls
    aiohttp_session = aiohttp.ClientSession()

    body_data = json.loads(body)
    phone_number = body_data["to_phone_number"]
    # caller_id = body_data["from_phone_number"]
    caller_id = ""

    # Simple state tracking
    current_mode = MUTE_MODE
    is_voicemail = False

    # Notifier for human conversation gate
    human_notifier = EventNotifier()

    # Observer for voicemail detection
    voicemail_observer = VoicemailDetectionObserver()

    # ------------ FUNCTION HANDLERS ------------

    async def voicemail_detected(params: FunctionCallParams):
        nonlocal current_mode, is_voicemail

        confidence = params.arguments["confidence"]
        reasoning = params.arguments["reasoning"]

        logger.info(
            f"Voicemail detected - confidence: {confidence}, reasoning: {reasoning}")

        if confidence >= VOICEMAIL_CONFIDENCE_THRESHOLD and current_mode == MUTE_MODE:
            current_mode = VOICEMAIL_MODE
            is_voicemail = True

            await voicemail_observer.wait_for_voicemail()

            # Get lead information from flow manager state
            lead_first_name = flow_manager.state.get(
                "lead_first_name", "there")
            agent_name = flow_manager.state.get("bot_name", "Kim")
            pf_program = flow_manager.state.get("pf_program", "our programs")

            # Generate Penn Foster voicemail message
            message = f"Hello {lead_first_name}, this is {agent_name} with Penn Foster calling to follow up on your request for information about our {pf_program} program. Please call us today at (855) 522-9232 for more info on this program. Thank you, and have a great day."
            await voicemail_tts.queue_frame(TTSSpeakFrame(text=message))
            await voicemail_tts.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

        await params.result_callback({"confidence": f"{confidence}", "reasoning": reasoning})

    async def human_detected(params: FunctionCallParams):
        nonlocal current_mode, is_voicemail

        confidence = params.arguments["confidence"]
        reasoning = params.arguments["reasoning"]

        logger.info(
            f"Human detected - confidence: {confidence}, reasoning: {reasoning}")

        if confidence >= HUMAN_CONFIDENCE_THRESHOLD and current_mode == MUTE_MODE:
            current_mode = HUMAN_MODE
            is_voicemail = False

            await human_notifier.notify()
            await flow_manager.initialize()
            # await flow_manager.set_node("greeting", create_greeting_node())

        await params.result_callback({"confidence": f"{confidence}", "reasoning": reasoning})

    # async def terminate_call(params: FunctionCallParams):
    #     logger.info("Terminating call")
    #     await asyncio.sleep(3)  # Brief delay before termination
    #     await params.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
    #     await params.result_callback({"status": "call terminated"})

    # ------------ TRANSPORT & SERVICES ------------

    transport = DailyTransport(
        room_url,
        token,
        "Voicemail Detection Bot",
        DailyParams(
            # api_url=daily_api_url,
            # api_key=daily_api_key,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=False,
        ),
    )

    # TTS services
    voicemail_tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="b7d50908-b17c-442d-ad8d-810c63997ed9",
    )

    human_tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="b7d50908-b17c-442d-ad8d-810c63997ed9",
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # ------------ LLM SETUP ------------

    detection_tools = [
        {
            "function_declarations": [
                {
                    "name": "voicemail_detected",
                    "description": "Signals that a voicemail greeting has been detected by the LLM.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "confidence": {
                                "type": "number",
                                "description": "The LLM's confidence score (ranging from 0.0 to 1.0) that a voicemail greeting was detected.",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "The LLM's textual explanation for why it believes a voicemail was detected, often citing specific phrases from the transcript.",
                            },
                        },
                        "required": ["confidence", "reasoning"],
                    },
                },
                {
                    "name": "human_detected",
                    "description": "Signals that a human attempting to communicate has been detected by the LLM.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "confidence": {
                                "type": "number",
                                "description": "The LLM's confidence score (ranging from 0.0 to 1.0) that a human conversation has been detected.",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "The LLM's textual explanation for why it believes a human communication was detected, often citing specific phrases from the transcript.",
                            },
                        },
                        "required": ["confidence", "reasoning"],
                    },
                },
            ]
        }
    ]

    # human_tools = [
    #     {"function_declarations": [{"name": "terminate_call", "description": "End the call"}]}
    # ]

    detection_system_instructions = """
    You are an AI Call Analyzer. Your primary function is to determine if the initial audio from an incoming call is a voicemail system/answering machine or a live human attempting to engage in conversation.

    You will be provided with a transcript of the first few seconds of an audio interaction.

    Based on your analysis of this transcript, you MUST decide to call ONE of the following two functions:

    1.  voicemail_detected
        *   Call this function if the transcript strongly indicates a pre-recorded voicemail greeting, an answering machine message, or instructions to leave a message.
        *   Keywords and phrases to look for: "you've reached," "not available," "leave a message," "at the tone/beep," "sorry I missed your call," "please leave your name and number."
        *   Also consider if the speech sounds like a monologue without expecting an immediate response.
        *   Keep in mind that the beep noise from a typical pre-recorded voicemail greeting comes after the greeting and not before.

    2.  human_detected
        *   Call this function if the transcript indicates a human is present and actively trying to communicate or expecting an immediate response.
        *   Keywords and phrases to look for: "Hello?", "Hi," "[Company Name], how can I help you?", "Speaking.", or any direct question aimed at initiating a dialogue.
        *   Consider if the speech sounds like the beginning of a two-way conversation.

    **Decision Guidelines:**

    *   **Prioritize Human:** If there's ambiguity but a slight indication of a human trying to speak (e.g., a simple "Hello?" followed by a pause, which could be either), err on the side of `human_detected` to avoid missing a live interaction. Only call `voicemail_detected` if there are clear, strong indicators of a voicemail system.
    *   **Focus on Intent:** Is the speaker *delivering information* (likely voicemail) or *seeking interaction* (likely human)?
    *   **Brevity:** Voicemail greetings are often concise and formulaic. Human openings can be more varied."""

    detection_llm = GoogleLLMService(
        model="models/gemini-2.0-flash-lite",
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=detection_system_instructions,
        tools=detection_tools,
    )

    human_llm = GoogleLLMService(
        model="models/gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        # system_instruction="""You are Chatbot talking to a human. Be friendly and helpful.
        # Start with: "Hello! I'm a friendly chatbot. How can I help you today?"
        # Keep your responses brief and to the point. Listen to what the person says.
        # If the user asks you to check the context, call the function `context_check`.
        # When the person indicates they're done with the conversation by saying something like:
        # - "Goodbye"
        # - "That's all"
        # - "I'm done"
        # - "Thank you, that's all I needed"
        # THEN say: "Thank you for chatting. Goodbye!" and call the terminate_call function.""",
        # tools=human_tools,
    )

    # ------------ CONTEXTS & FUNCTIONS ------------

    # context = GoogleLLMContext()
    # context_aggregator = detection_llm.create_context_aggregator(context)

    detection_context = GoogleLLMContext()
    detection_context_aggregator = detection_llm.create_context_aggregator(
        detection_context)

    human_context = GoogleLLMContext()
    human_context_aggregator = human_llm.create_context_aggregator(
        human_context)

    # Register functions
    detection_llm.register_function("voicemail_detected", voicemail_detected)
    detection_llm.register_function("human_detected", human_detected)
    # human_llm.register_function("terminate_call", terminate_call)

    # ------------ PROCESSORS ------------

    audio_collector = UserAudioCollector(
        detection_context, detection_context_aggregator.user())
    human_gate = OutputGate(human_notifier, start_open=False)

    # Filter functions
    async def voicemail_filter(frame) -> bool:
        return current_mode == VOICEMAIL_MODE

    async def human_filter(frame) -> bool:
        return current_mode == HUMAN_MODE

    # ------------ PIPELINE ------------

    pipeline = Pipeline(
        [
            transport.input(),
            ParallelPipeline(
                # Voicemail detection branch
                [
                    audio_collector,
                    detection_context_aggregator.user(),
                    detection_llm,
                    voicemail_tts,
                    FunctionFilter(voicemail_filter),
                ],
                # Human conversation branch
                [
                    stt,
                    human_context_aggregator.user(),
                    human_llm,
                    human_gate,
                    human_tts,
                    FunctionFilter(human_filter),
                    human_context_aggregator.assistant(),
                ],
            ),
            transport.output(),
        ]
    )

    pipeline_task = PipelineTask(
        pipeline,
        idle_timeout_secs=90,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        cancel_on_idle_timeout=False,
        observers=[voicemail_observer],
    )

    flow_manager = FlowManager(
        task=pipeline_task,
        tts=human_tts,
        llm=human_llm,
        context_aggregator=human_context_aggregator,
        transport=transport,
    )

    script_info = body_data.get("scriptInfo", {})

    # Initialize flow manager state with all necessary data
    flow_manager.state.update({
        "bot_name": "Kim",
        "escalation_number": body_data.get("escalation_number"),
        "partner_phone_number": body_data.get("partner_phone_number"),
        "session_id": body_data.get("session_id"),
        "from_phone_number": body_data.get("from_phone_number"),
        "to_phone_number": phone_number,
        "daily_room_url": body_data.get("daily_room_url"),
        "music_token": body_data.get("music_token"),
        "lead_id": body_data.get("lead_id"),
        "program_id": body_data.get("program_id"),
        "event_log_id": body_data.get("event_log_id"),
        # Script Info
        "lead_first_name": script_info.get("lead_first_name"),
        "lead_state": script_info.get("lead_state"),
        "pf_program": script_info.get("pf_program"),
    })

    # await flow_manager.initialize()

    # ------------ EVENT HANDLERS ------------

    @transport.event_handler("on_joined")
    async def on_joined(transport, data):
        if not use_prebuilt:
            dialout_params = {"phoneNumber": phone_number}
            if caller_id:
                dialout_params["callerId"] = caller_id
            logger.info(
                f"Dialing out to {phone_number} with caller ID {caller_id}")
            await transport.start_dialout(dialout_params)

    @transport.event_handler("on_dialout_answered")
    async def on_dialout_answered(transport, data):
        logger.debug(f"Call answered: {data}")
        await transport.capture_participant_transcription(data["sessionId"])

    @transport.event_handler("on_dialout_error")
    async def on_dialout_error(transport, data):
        logger.error(f"Dialout error: {data}")
        await pipeline_task.queue_frame(EndFrame())

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await pipeline_task.queue_frame(EndFrame())

    # Remove the problematic on_pipeline_started handler
    # The context will be initialized naturally when frames flow through the pipeline

    # ------------ RUN ------------

    runner = PipelineRunner()
    logger.info("Starting simplified parallel pipeline bot")

    try:
        await runner.run(pipeline_task)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback

        logger.error(traceback.format_exc())
    finally:
        await aiohttp_session.close()


# ------------ ENTRY POINT ------------

async def bot(args: DailySessionArguments) -> None:
    """Main Entry point for Pipecat Cloud Bot"""
    args_body = args.body
    await run_bot(args_body["daily_room_url"],
                  args_body["bot_token"], json.dumps(args_body))
    # session_id
    # daily_room_url
    # event_log_id
    # lead_id
    # program_id
    # bot_token
    # music_token
    # from_phone_number
    # to_phone_number
    # scriptInfo {}
    # escalation_number
    # partner_phone_number


async def create_room_and_tokens(api_key: str) -> tuple[str, str, str]:
    """Create a Daily room and generate bot and music tokens.

    Args:
        api_key: Daily API key

    Returns:
        Tuple of (room_url, bot_token, music_token)
    """
    if not api_key:
        raise ValueError("Daily API key is required")

    api_url = "https://api.daily.co/v1"

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


async def run_local_test() -> None:
    """Run a local test of the bot."""
    # Create a room
    room_url, bot_token, music_token = await create_room_and_tokens(daily_api_key)
    # Create a test body
    mock_body = {
        "daily_room_url": room_url,
        "bot_token": bot_token,
        "music_token": music_token,
        "session_id": "test-session",
        "event_log_id": 1234,
        "lead_id": 12345,
        "program_id": 67890,
        "from_phone_number": os.getenv("FROM_PHONE_NUMBER"),
        "to_phone_number": os.getenv("TO_PHONE_NUMBER"),
        "scriptInfo": {
            "lead_first_name": "John",
            "lead_state": "California",
            "pf_program": "High School",
        },
        "escalation_number": os.getenv("ESCALATION_NUMBER"),
        "partner_phone_number": os.getenv("PARTNER_PHONE_NUMBER"),
    }

    await run_bot(
        room_url=room_url,
        token=bot_token,
        body=json.dumps(mock_body)
    )


if __name__ == "__main__":
    asyncio.run(run_local_test())
