# script_pages/consumer_dial_back.py

from typing import Dict, Any
from loguru import logger
from pipecat.transports.services.daily import DailyTransport

from pipecat_flows import FlowManager, FlowsFunctionSchema, NodeConfig
from pipecat_flows.types import ActionConfig


# --- Consumer Dial Back: Handle consumer answering the reconnect call ---


async def handle_consumer_dialback_answered(args: Dict[str, Any], flow_manager: "FlowManager") -> Dict[str, Any]:
    """
    Handles the consumer answering the dialback call.
    Sets up state for voicemail detection and consumer conversation flow.
    """
    logger.info(
        "Consumer dialback answered - preparing for voicemail detection or live conversation")

    # Set state to indicate we're in consumer dialback mode
    flow_manager.state["in_consumer_dialback"] = True
    flow_manager.state["voicemail_detection_active"] = True
    # You might want to use actual timestamp
    flow_manager.state["dialback_start_time"] = "now"

    # Clear any previous partner-related state
    flow_manager.state.pop("partner_participant_id", None)

    return {"status": "consumer_dialback_answered"}


async def detect_voicemail_or_live_answer(args: Dict[str, Any], flow_manager: "FlowManager") -> Dict[str, Any]:
    """
    Function to call when we need to determine if this is voicemail or live answer.
    This is called by the LLM after analyzing the initial audio/response.
    """
    detection_type = args.get("detection_type", "").lower().strip()
    user_response = args.get("user_response", "").lower().strip()

    logger.info(
        f"Voicemail detection called with type: '{detection_type}', response: '{user_response}'")

    if detection_type == "voicemail":
        logger.info("Voicemail system detected by LLM analysis")
        flow_manager.state["voicemail_detected"] = True
        flow_manager.state["disposition"] = "Warm transfer attempted, consumer hung up during transfer"
        return {"status": "voicemail_detected", "transition_to": "voicemail_detected_node"}
    elif detection_type == "human":
        logger.info("Live human detected by LLM analysis")
        flow_manager.state["voicemail_detected"] = False
        flow_manager.state["live_consumer_reconnected"] = True
        return {"status": "live_consumer_detected", "transition_to": "live_consumer_reconnected_node"}
    else:
        # Default to voicemail if unclear
        logger.warning(
            f"Unclear detection result, defaulting to voicemail: {detection_type}")
        flow_manager.state["voicemail_detected"] = True
        flow_manager.state["disposition"] = "Warm transfer attempted, consumer hung up during transfer"
        return {"status": "voicemail_detected", "transition_to": "voicemail_detected_node"}


async def handle_consumer_response_to_reconnect(args: Dict[str, Any], flow_manager: "FlowManager") -> Dict[str, Any]:
    """
    Handle the consumer's response to whether they still want to talk to a specialist.
    """
    user_response = args.get("user_response", "").lower().strip()

    logger.info(f"Consumer responded to reconnect question: '{user_response}'")

    # Check for positive responses
    if any(word in user_response for word in ["yes", "yeah", "yep", "sure", "okay", "ok", "still", "want"]):
        logger.info(
            "Consumer wants to continue with transfer - initiating partner callback")
        flow_manager.state["consumer_wants_transfer"] = True
        return {"status": "initiate_transfer", "transition_to": "initiate_partner_callback_node"}
    else:
        logger.info("Consumer does not want to continue - ending call")
        flow_manager.state["consumer_wants_transfer"] = False
        flow_manager.state["disposition"] = "Consumer reconnected but declined transfer"
        return {"status": "end_call", "transition_to": "end_call_node"}


def create_consumer_dialback_greeting_node(flow_manager: "FlowManager") -> NodeConfig:
    """
    Initial greeting when consumer answers the dialback call.
    Simple message about getting disconnected.
    """
    lead_first_name = flow_manager.state.get("lead_first_name", "there")

    simple_message = f"Hi {lead_first_name}! We got disconnected during our call."

    return NodeConfig(
        id="consumer_dialback_greeting_node",
        task_messages=[{"role": "assistant", "content": simple_message}],
        functions=[],
        respond_immediately=True,
        post_actions=[]
    )


def create_voicemail_detected_node(flow_manager: "FlowManager") -> NodeConfig:
    """
    Node for when voicemail is detected during consumer dialback.
    Leaves a professional message.
    """
    agent_name = flow_manager.state.get("agent_name", "Credit Associates")

    voicemail_script = (f"Hi this is {agent_name} from Credit Associates. "
                        f"I'm sorry that we were disconnected. "
                        f"Please call back at (855) 522-9232. Thanks!")

    flow_manager.state["last_bot_utterance"] = voicemail_script
    flow_manager.state["disposition"] = "Warm transfer attempted, consumer hung up during transfer"

    # Mark call as complete after leaving voicemail
    flow_manager.state["call_completed"] = True

    return NodeConfig(
        id="voicemail_detected_node",
        task_messages=[{"role": "assistant", "content": voicemail_script}],
        functions=[],
        respond_immediately=True
    )


async def initiate_partner_callback(args: Dict[str, Any], flow_manager: "FlowManager") -> Dict[str, Any]:
    """
    Initiates a callback to the partner to complete the transfer.
    This is called when the consumer has been successfully reconnected.
    """
    partner_phone_number = flow_manager.state.get("partner_phone_number")
    lead_first_name = flow_manager.state.get("lead_first_name", "the consumer")

    if not partner_phone_number:
        logger.error("Partner phone number not found for callback")
        flow_manager.state["disposition"] = "Consumer reconnected but partner callback failed - no partner number"
        return {"status": "partner_callback_failed_no_number"}

    try:
        transport = flow_manager.transport
        await transport.start_dialout({
            "phoneNumber": partner_phone_number,
            "displayName": 'partner',
            "userId": 'partner',
            "permissions": {
                "canReceive": {"base": False, "byUserId": {"bot": True}},
            }
        })

        logger.info(
            f"Partner callback initiated for {lead_first_name} transfer completion")
        flow_manager.state["partner_callback_initiated"] = True
        flow_manager.state["disposition"] = "Consumer reconnected, partner callback in progress"
        return {"status": "partner_callback_initiated"}

    except Exception as e:
        logger.error(f"Partner callback failed: {e}")
        flow_manager.state["disposition"] = "Consumer reconnected but partner callback failed - dialout error"
        return {"status": "partner_callback_failed_dialout_error"}


async def mute_customer_dialback(args: Dict[str, Any], flow_manager: "FlowManager") -> Dict[str, Any]:
    """Mute the customer during dialback transfer process."""
    transport = flow_manager.transport
    customer_participant_id = flow_manager.state.get('consumer_participant_id')
    flow_manager.state["second_transfer_attempt"] = True

    logger.debug(
        f"DIALBACK: Muting customer with ID {customer_participant_id}")

    if transport and customer_participant_id:
        try:
            await transport.update_remote_participants(
                remote_participants={
                    customer_participant_id: {
                        "permissions": {
                            "canSend": [],
                        }
                    }
                }
            )
            logger.info(
                f"DIALBACK: Successfully muted customer {customer_participant_id}")
        except Exception as e:
            logger.error(f"DIALBACK: Failed to mute customer: {e}")
    else:
        logger.error(f"DIALBACK: Missing transport or customer_participant_id")

    return {"status": "customer_muted"}


async def start_hold_music_dialback(args: Dict[str, Any], flow_manager: "FlowManager") -> Dict[str, Any]:
    """Start hold music for the consumer during dialback transfer."""
    try:
        daily_room_url = flow_manager.state.get('daily_room_url')
        music_token = flow_manager.state.get('music_token')
        hold_music_manager = flow_manager.state.get('hold_music_manager')

        success = await hold_music_manager.start(daily_room_url, music_token)

        if success:
            logger.info("DIALBACK: Hold music started successfully")
            return {"status": "hold_music_started"}
        else:
            logger.error("DIALBACK: Failed to start hold music")
            return {"status": "hold_music_failed"}

    except Exception as e:
        logger.error(f"DIALBACK: Error starting hold music: {e}")
        return {"status": "hold_music_error"}


async def make_customer_hear_only_hold_music_dialback(args: Dict[str, Any], flow_manager: "FlowManager") -> Dict[str, Any]:
    """Make customer only hear hold music during dialback."""
    transport: DailyTransport = flow_manager.transport
    customer_participant_id = flow_manager.state.get('consumer_participant_id')

    logger.debug(
        f"DIALBACK: Setting customer {customer_participant_id} to hear only hold music")

    try:
        await transport.update_remote_participants(
            remote_participants={
                customer_participant_id: {
                    "permissions": {
                        "canReceive": {
                            "base": False,
                            "byUserId": {"hold-music": True},
                        }
                    },
                }
            }
        )
        logger.info(
            f"DIALBACK: Successfully set customer {customer_participant_id} to hear only hold music")
        return {"status": "hold_music_permissions_set"}
    except Exception as e:
        logger.error(
            f"DIALBACK: Failed to set customer hold music permissions: {e}")
        return {"status": "hold_music_permissions_failed"}


def create_live_consumer_reconnected_node(flow_manager: "FlowManager") -> NodeConfig:
    """
    Node for when live consumer is detected during dialback.
    Explains the situation and initiates the same transfer process as page 5/6.
    """
    lead_first_name = flow_manager.state.get("lead_first_name", "there")
    partner_name = flow_manager.state.get("partner_name", "our partner")

    reconnect_script = (f"Hi {lead_first_name}! Great to have you back. "
                        f"We got disconnected while I was transferring you to {partner_name}. "
                        f"Let me reconnect you now. Please hold for just a moment.")

    flow_manager.state["last_bot_utterance"] = reconnect_script
    flow_manager.state["ready_for_partner_callback"] = True

    task_content = f"""Say "{reconnect_script}" and then stop speaking completely.

After you finish speaking, the system will automatically:
1. Mute the customer
2. Start hold music for them
3. Make them hear only hold music (not your conversation with the specialist)
4. Initiate the warm transfer to reconnect the specialist

CRITICAL INSTRUCTIONS:
- Say ONLY the script above, nothing more
- Do NOT call any functions yourself - they will be handled automatically
- Do NOT speak again after delivering the message
- Do NOT ask follow-up questions
- Do NOT check if the user is still there
- Do NOT generate any additional audio output
- The transfer process will begin automatically once you finish speaking
- You will remain silent until the specialist connects and you receive new system instructions

The user will be placed on hold with music while the specialist connection is re-established."""

    return NodeConfig(
        id="live_consumer_reconnected_node",
        task_messages=[{"role": "system", "content": task_content}],
        functions=[],
        pre_actions=[
            ActionConfig(type="function", handler=mute_customer_dialback),
        ],
        post_actions=[
            ActionConfig(type="function", handler=start_hold_music_dialback),
            ActionConfig(type="function",
                         handler=make_customer_hear_only_hold_music_dialback),
            ActionConfig(type="function", handler=initiate_partner_callback)
        ]
    )


async def handle_consumer_dialback_timeout(args: Dict[str, Any], flow_manager: "FlowManager") -> Dict[str, Any]:
    """
    Handle case where consumer doesn't answer the dialback within reasonable time.
    """
    logger.info("Consumer dialback timeout - treating as voicemail")
    flow_manager.state["voicemail_detected"] = True
    flow_manager.state["disposition"] = "Warm transfer attempted, consumer hung up during transfer"

    return {"status": "dialback_timeout"}


def create_consumer_dialback_timeout_node(flow_manager: "FlowManager") -> NodeConfig:
    """
    Node for handling consumer dialback timeout scenarios.
    """
    return NodeConfig(
        id="consumer_dialback_timeout_node",
        task_messages=[],
        functions=[],
        respond_immediately=True,
        post_actions=[ActionConfig(
            type="function",
            handler=handle_consumer_dialback_timeout
        )]
    )


def create_initiate_partner_callback_node(flow_manager: "FlowManager") -> NodeConfig:
    """
    Node for when consumer wants to continue with transfer - initiates partner callback.
    """
    reconnect_script = "Great! Let me reconnect you now. Please hold for just a moment."

    return NodeConfig(
        id="initiate_partner_callback_node",
        task_messages=[{"role": "assistant", "content": reconnect_script}],
        functions=[],
        respond_immediately=True,
        post_actions=[
            ActionConfig(type="function", handler=mute_customer_dialback),
            ActionConfig(type="function", handler=start_hold_music_dialback),
            ActionConfig(type="function",
                         handler=make_customer_hear_only_hold_music_dialback),
            ActionConfig(type="function", handler=initiate_partner_callback)
        ]
    )


def create_end_call_node(flow_manager: "FlowManager") -> NodeConfig:
    """
    Node for when consumer doesn't want to continue - ends the call politely.
    """
    end_call_script = "I understand. Thank you for your time today. Have a great day!"

    flow_manager.state["call_completed"] = True

    return NodeConfig(
        id="end_call_node",
        task_messages=[{"role": "assistant", "content": end_call_script}],
        functions=[],
        respond_immediately=True,
        post_actions=[
            ActionConfig(type="end_conversation")
        ]
    )
