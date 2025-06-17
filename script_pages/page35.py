# script_pages/page35.py

from loguru import logger

# Added ContextStrategy, ContextStrategyConfig
from pipecat_flows import FlowManager, NodeConfig, ContextStrategy, ContextStrategyConfig
from pipecat.transports.services.daily import DailyTransport
from pipecat_flows.types import ActionConfig


async def unmute_consumer_and_have_bot_back(action: dict, flow_manager: FlowManager):
    """Unmute the consumer and make it so the customer and bot can hear each other."""
    transport: DailyTransport = flow_manager.transport
    consumer_participant_id = flow_manager.state.get("consumer_participant_id")

    await transport.update_remote_participants(
        remote_participants={
            consumer_participant_id: {
                "permissions": {
                    "canSend": ["microphone"],
                    "canReceive": {"byUserId": {"bot": True}},
                },
                "inputsEnabled": {"microphone": True},
            }
        }
    )
    logger.info("Consumer unmuted and bot audio re-enabled for consumer.")

# --- Helper action function ---


async def _action_stop_dialout(action: dict, flow_manager: FlowManager):
    """Custom action to stop dialout to a participant."""
    logger.info("PAGE35: Preparing to stop dialout.")

    transport: DailyTransport = flow_manager.transport
    partner_participant_id = flow_manager.state.get("partner_participant_id")
    if partner_participant_id:
        logger.info(f"Stopping dialout for partner: {partner_participant_id}")
        try:
            await transport.stop_dialout(participant_id=partner_participant_id)
            logger.info(
                f"Successfully stopped dialout for partner: {partner_participant_id}")
        except Exception as e:
            logger.error(
                f"Error stopping dialout for partner {partner_participant_id}: {e}")
    else:
        logger.error(
            "Partner participant ID not provided for stop_dialout action.")


async def _action_stop_consumer_dialout(action: dict, flow_manager: FlowManager):
    """Custom action to stop dialout to the consumer."""
    logger.info("PAGE35: Preparing to stop dialout for consumer.")

    transport: DailyTransport = flow_manager.transport
    consumer_participant_id = flow_manager.state.get("consumer_participant_id")
    if consumer_participant_id:
        logger.info(
            f"Stopping dialout for consumer: {consumer_participant_id}")
        try:
            await transport.stop_dialout(participant_id=consumer_participant_id)
            logger.info(
                f"Successfully stopped dialout for consumer: {consumer_participant_id}")
        except Exception as e:
            logger.error(
                f"Error stopping dialout for consumer {consumer_participant_id}: {e}")
    else:
        logger.error(
            "Consumer participant ID not provided for stop_dialout action.")


# --- Node 1: Speak to Partner, Hangup Partner, Unmute Consumer ---

# Define the transition handler at module level or ensure it's correctly scoped if nested.
async def transition_to_inform_user_node_action(action_cfg: dict, fm: FlowManager):
    """Action handler to explicitly transition to the inform user node."""
    logger.debug("Executing transition to page_35_inform_user_node")
    await fm.set_node("page_35_inform_user_node", create_page_35_inform_user_node(fm))


def create_page_35_speak_to_partner_node(flow_manager: "FlowManager") -> NodeConfig:
    """
    Node 1: Notifies the specialist, hangs up on them, unutes consumer, then transitions.
    """
    logger.warning(
        "PAGE 35: Specialist refused warm transfer. Notifying specialist.")

    partner_participant_id = flow_manager.state.get("partner_participant_id")

    partner_script = "Ok, thank you. I will inform the consumer that you are unable to assist them at this time. Have a good day."
    # This user_script is for logging context here, actual speaking is in the next node.
    user_script_for_context = ("I'm very sorry, but we were unable to connect you with a specialist at this time. "
                               "We apologize for the inconvenience. Thank you for your time. Goodbye.")

    # Task for the LLM: only speak to the specialist.
    task_content = f"""This is a phone conversation and your responses will be converted to audio. Keep the conversation friendly, casual, and polite. Avoid outputting special characters and emojis.
Your only task is to speak the following message to the specialist. Do not add any other commentary or phrases. Message: "{partner_script}" """

    return NodeConfig(
        id="page_35_speak_to_partner_node",
        task_messages=[{"role": "system", "content": task_content}],
        respond_immediately=True,
        post_actions=[ActionConfig(type="function", handler=_action_stop_dialout),
                      ActionConfig(type="function",
                                   handler=unmute_consumer_and_have_bot_back, ),
                      ActionConfig(type="function",  # Changed from "transition"
                                   handler=transition_to_inform_user_node_action)],
        functions=[]
    )

# --- Node 2: Inform User and End Conversation ---


def create_page_35_inform_user_node(flow_manager: "FlowManager") -> NodeConfig:
    """
    Node 2: Informs the user that the transfer could not be completed and ends the call.
    """
    logger.info("PAGE 35: Informing user about inability to connect specialist.")

    user_script = ("I'm very sorry, but we were unable to connect you with a specialist at this time. "
                   "We apologize for the inconvenience. Thank you for your time. Goodbye.")

    task_content = f"""This is a phone conversation and your responses will be converted to audio. Keep the conversation friendly, casual, and polite. Avoid outputting special characters and emojis.
Your only task is to speak the following message to the user. Do not add any other commentary or phrases. Message: "{user_script}" """

    return NodeConfig(
        id="page_35_inform_user_node",
        task_messages=[{"role": "system", "content": task_content}],
        respond_immediately=True,  # Speak to user immediately
        context_strategy=ContextStrategyConfig(
            strategy=ContextStrategy.RESET),  # Added context strategy reset
        # End call after speaking
        post_actions=[
            # New action to hang up consumer
            ActionConfig(type="function",
                         handler=_action_stop_consumer_dialout),
            ActionConfig(type="end_conversation")
        ],
        functions=[]
    )

# --- Original Entry Node for Page 35 ---


def create_page_35_entry_node(flow_manager: "FlowManager") -> NodeConfig:
    """
    This is the entry point for page 35 logic when a specialist refuses a warm transfer.
    It initiates a sequence: notify specialist, hang up specialist, unmute consumer, notify consumer, hang up.
    """
    # The flow now starts with speaking to the partner.
    return create_page_35_speak_to_partner_node(flow_manager)
