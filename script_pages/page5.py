# script_pages/page5.py

from typing import Dict, Any
from loguru import logger

from pipecat_flows import FlowManager, FlowsFunctionSchema, NodeConfig
from pipecat_flows.types import ActionConfig,  FlowsFunctionSchema
from pipecat.transports.services.daily import DailyTransport


# --- Page 5 Response & Outcome Categories ---


CALLBACK_NUMBER_RESPONSE = ["yes", "no"]

TRANSFER_OUTCOME = [
    "specialist_answered",
    "on_hold_too_long",
    "just_rings",
    "busy_signal",
    "phone_out_of_order",
    "voicemail",
    "consumer_hung_up_after_partner",
    "other"
]


# STEP 1: Logic for confirming the callback number


async def confirm_number_handler(args: Dict[str, Any], flow_manager: "FlowManager") -> Dict[str, Any]:
    """Logs the user's response about their callback number."""
    response = args.get("user_response", "no")

    # Get the greeting line and user response for logging
    greeting_line = flow_manager.state.get("last_greeting_line", "")
    user_message = flow_manager.state.get("last_user_message", "")

    flow_manager.state["initate_transfer"] = True

    logger.info(
        f"PAGE 5 HANDLER: User confirmation for callback number: {response}")
    return {"status": "proceed", "data": args}


async def handle_number_confirmation_transition(args: Dict, result: Dict[str, Any], flow_manager: "FlowManager"):
    """Transitions to the transfer initiation node after user confirms their number."""
    logger.info(
        "PAGE 5 TRANSITION: Number confirmed, proceeding to initiate transfer.")
    await flow_manager.set_node("page_5_initiate_transfer_node", create_page_5_initiate_transfer_node(flow_manager))


async def mute_customer(action: dict, flow_manager: FlowManager):
    """Mute the customer.

    Do it by revoking their canSend permission, which both mutes them and ensures that they can't unmute.
    """

    transport: DailyTransport = flow_manager.transport
    customer_participant_id = flow_manager.state.get('consumer_participant_id')

    logger.debug(
        f"PAGE 5 MUTE: Muting customer with ID {customer_participant_id}")

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
                f"PAGE 5 MUTE: Successfully muted customer {customer_participant_id}")
        except Exception as e:
            logger.error(f"PAGE 5 MUTE: Failed to mute customer: {e}")
    else:
        logger.error(
            f"PAGE 5 MUTE: Missing transport ({transport}) or customer_participant_id ({customer_participant_id})")


async def start_hold_music(action: dict, flow_manager: FlowManager):
    """Start hold music for the consumer during transfer."""
    try:
        # Get room URL and token from transport
        daily_room_url = flow_manager.state.get('daily_room_url')
        music_token = flow_manager.state.get('music_token')
        hold_music_manager = flow_manager.state.get('hold_music_manager')

        success = await hold_music_manager.start(daily_room_url, music_token)

        if success:
            # Store the manager in state for later cleanup
            logger.info("Hold music started successfully")
        else:
            logger.error("Failed to start hold music")

    except Exception as e:
        logger.error(f"Error starting hold music: {e}")


async def stop_hold_music(action: dict, flow_manager: FlowManager):
    """Stop hold music when transfer is complete."""
    try:
        hold_music_manager = flow_manager.state.get("hold_music_manager")
        if hold_music_manager:
            await hold_music_manager.stop()
            # Clean up the reference
            flow_manager.state.pop("hold_music_manager", None)
            logger.info("Hold music stopped successfully")
        else:
            logger.debug("No hold music manager found to stop")
    except Exception as e:
        logger.error(f"Error stopping hold music: {e}")


async def make_customer_hear_only_hold_music(action: dict, flow_manager: FlowManager):
    """Make it so the customer only hears hold music.

    We don't want them hearing the bot and the human agent talking.
    """
    transport: DailyTransport = flow_manager.transport

    customer_participant_id = flow_manager.state.get('consumer_participant_id')

    logger.debug(
        f"make_customer_hear_only_hold_music: {customer_participant_id}")

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
            f"Successfully set customer {customer_participant_id} to hear only hold music")
    except Exception as e:
        logger.error(f"Failed to set customer hold music permissions: {e}")


async def start_warm_transfer(action: dict, flow_manager: FlowManager):
    """Start the warm transfer process.

    This is a stub for the global agent transfer request.
    """
    logger.info("PAGE 5 HANDLER: Initiating warm transfer...")

    transport: DailyTransport = flow_manager.transport

    # Start the partner dialout
    try:
        partner_phone_number = flow_manager.state.get(
            'partner_phone_number')
        logger.info(
            f"PAGE 5 HANDLER: Starting partner dialout to {partner_phone_number}")
        await transport.start_dialout({
            "phoneNumber": "+19519887911",
            "displayName": "partner",
            "userId": "partner",
            "permissions": {
                "canReceive": {
                    "base": False,
                      "byUserId": {
                          "bot": True,
                      }
                }
            }
        })
        logger.info("PAGE 5 HANDLER: Partner dialout started successfully")

    except Exception as e:
        logger.error(
            f"PAGE 5 HANDLER: Failed to start partner dialout: {e}")
        return {
            "status": "error",
            "message": "Failed to connect to specialist",
            "data": {"transfer_outcome": "other"}
        }

    logger.info(
        "PAGE 5 HANDLER: Partner dialout initiated, waiting for partner to answer...")

    # Get the greeting line and user response for logging
    greeting_line = flow_manager.state.get("last_greeting_line", "")
    user_message = flow_manager.state.get("current_user_message", "")
    flow_manager.state["last_user_message"] = user_message


# --- Unified Nodes for Page 5 ---


def create_page_5_initiate_transfer_node(flow_manager: "FlowManager") -> NodeConfig:
    """UNIFIED NODE 2: Speaks the 'please hold' message and immediately calls the transfer function."""
    # Store the user's response in state when it comes in
    flow_manager.state["last_user_message"] = flow_manager.state.get(
        "current_user_message", "")

    lead_first_name = flow_manager.state.get("lead_first_name", "")
    bot_script = f"Thanks. Please hold with me {lead_first_name}"

    # Store the greeting line for logging
    flow_manager.state["last_greeting_line"] = bot_script

    task_content = f"""Say "{bot_script}" and then stop speaking completely.

After you finish speaking, the system will automatically:
1. Mute the customer
2. Start hold music for them
3. Make them hear only hold music (not your conversation with the specialist)
4. Initiate the warm transfer to connect the specialist

CRITICAL INSTRUCTIONS:
- Say ONLY the script above, nothing more
- Do NOT call any functions yourself - they will be handled automatically
- Do NOT speak again after delivering the message
- Do NOT ask follow-up questions
- Do NOT check if the user is still there
- Do NOT generate any additional audio output
- The transfer process will begin automatically once you finish speaking
- You will remain silent until the specialist connects and you receive new system instructions

The user will be placed on hold with music while the specialist connection is established."""
    return NodeConfig(
        id="page_5_initiate_transfer_node",
        task_messages=[{"role": "system", "content": task_content}],
        functions=[
        ],
        pre_actions=[
            ActionConfig(type="function", handler=mute_customer),
        ],
        post_actions=[
            ActionConfig(type="function", handler=start_hold_music),
            ActionConfig(type="function",
                              handler=make_customer_hear_only_hold_music),
            ActionConfig(type="function", handler=start_warm_transfer)

        ]
    )


def create_page_5_entry_node(flow_manager: "FlowManager") -> NodeConfig:
    """UNIFIED NODE 1: Asks the user to confirm their callback number."""
    lead_first_name = flow_manager.state.get("lead_first_name", "the consumer")
    bot_script = f"Thanks! {lead_first_name} I'm going to place you on hold while I connect, then I will bring you back on the line and introduce you. This will only take a moment. Is this phone number the best way to reach you just in case we're disconnected?"

    # Store the greeting line for logging
    flow_manager.state["last_greeting_line"] = bot_script

    task_content = f"""This is a phone conversation and your responses will be converted to audio. Keep the conversation friendly, casual, and polite. Avoid outputting special characters and emojis.

Your task is to:
1.  Speak the following message to the user EXACTLY: "{bot_script}"
2.  After you have spoken the ENTIRE message, WAIT for the user to respond.
3.  Once the user has responded, call the `confirm_callback_number` function with their actual spoken response.

CRITICAL INSTRUCTIONS:
-   Speak the ENTIRE message first.
-   Do NOT interrupt yourself.
-   Do NOT call any functions until AFTER the user has spoken their response to your question.
-   Pass the user's literal response to the `confirm_callback_number` function."""

    return NodeConfig(
        id="page_5_entry_node",
        task_messages=[{"role": "system", "content": task_content}],
        functions=[
            FlowsFunctionSchema(
                name="confirm_callback_number",
                handler=confirm_number_handler,
                transition_callback=handle_number_confirmation_transition,
                description="Confirms if the current phone number is the best for a callback.",
                properties={
                    "user_response": {
                        "type": "string",
                        "description": "User's confirmation, 'yes' or 'no'.",
                        "enum": CALLBACK_NUMBER_RESPONSE
                    }
                },
                required=["user_response"]
            )
        ],
        interruptions_enabled=False,
        respond_immediately=True,  # Make sure bot speaks right away
    )
