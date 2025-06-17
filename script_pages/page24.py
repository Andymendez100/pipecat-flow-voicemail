# script_pages/page24.py

from typing import Dict, Any
from loguru import logger

from pipecat_flows import FlowManager,  NodeConfig
from pipecat_flows.types import ActionConfig

from pipecat.transports.base_transport import BaseTransport

# --- Page 24: Consumer Disconnected, Inform Partner, Hang Up Partner, Attempt Reconnect Consumer ---


async def initiate_consumer_reconnect_after_partner_goodbye(args: Dict[str, Any], flow_manager: "FlowManager") -> Dict[str, Any]:
    """
    1. Hangs up the partner's call leg.
    2. Attempts to reconnect with the consumer by initiating a dialout.
    The outcome of the consumer dialout will be handled by event handlers in bot.py.
    """
    transport: BaseTransport = flow_manager.transport
    lead_first_name = flow_manager.state.get("lead_first_name", "the consumer")
    partner_participant_id = flow_manager.state.get("partner_participant_id")

    # Step 1: Hang up the partner's call leg
    if partner_participant_id:
        try:
            logger.info(
                f"Page 24: Attempting to hang up partner leg: {partner_participant_id}")
            await transport.stop_dialout(participant_id=partner_participant_id)
            logger.info(
                f"Page 24: Successfully hung up partner leg: {partner_participant_id}")
        except Exception as e:
            logger.error(
                f"Page 24: Failed to hang up partner leg {partner_participant_id}: {e}")
            # Continue to attempt consumer reconnect even if partner hangup failed,
            # as the primary goal is to get the consumer back.
    else:
        logger.warning(
            "Page 24: partner_participant_id not found in state. Cannot hang up partner leg specifically.")
        # If no specific partner ID, we can't target them. The call might end naturally if no other participants.

    # Step 2: Attempt to reconnect with the consumer
    consumer_phone_number = flow_manager.state.get(
        "to_phone_number")  # Consumer's direct line

    if not consumer_phone_number:
        logger.error(
            "Page 24 Reconnect: Consumer phone number (to_phone_number) not found in state.")
        flow_manager.state["reconnect_attempt_outcome"] = "error_no_consumer_number"
        flow_manager.state["disposition"] = "Warm transfer attempted, consumer hung up, no consumer phone for reconnect"
        # No specific node to transition to here, bot.py will need to handle lack of further events.
        return {"status": "reconnect_failed_no_consumer_number"}

    try:
        await transport.start_dialout({
            "phoneNumber": consumer_phone_number,
            # "callerId": bot_outbound_number,
            "displayName": 'consumer',
            "userId": 'consumer',
            "permissions": {
                "canReceive": {"base": False, "byUserId": {"bot": True}},
            }
        })
        logger.info(
            f"Page 24: Consumer reconnect dialout initiated for {lead_first_name}.")
        flow_manager.state["reconnect_dialout_initiated"] = True
        return {"status": "consumer_reconnect_dialout_initiated"}
    except Exception as e:
        logger.error(
            f"Page 24: Consumer reconnect transport.start_dialout failed: {e}")
        flow_manager.state["reconnect_attempt_outcome"] = "error_dialout_exception"
        flow_manager.state["disposition"] = "Warm transfer attempted, consumer hung up, consumer reconnect dialout error"
        return {"status": "reconnect_failed_dialout_exception"}


def create_page_24_entry_node(flow_manager: "FlowManager") -> NodeConfig:
    """
    Entry node for Page 24.
    Informs the partner about the consumer's disconnection, that we will attempt to reconnect
    and call them back, says goodbye. The function called will then hang up the partner
    and initiate the consumer reconnect attempt.
    """

    flow_manager.state["dialback_consumer"] = True
    lead_first_name = flow_manager.state.get("lead_first_name", "the consumer")

    script_to_partner = (f"I apologize, {lead_first_name} did just disconnect. "
                         f"I will attempt to reconnect with them. If successful, I will call you back to complete the transfer. Goodbye for now.")

    flow_manager.state["last_bot_utterance"] = script_to_partner

    task_content = f"""
    You are speaking with a business partner. The consumer you were trying to transfer to them has disconnected.
    Your goal is to:
    1.  Inform the partner that the consumer disconnected.
    2.  Tell them you will attempt to reconnect with the consumer.
    3.  Tell them that IF the reconnect is successful, you will call the partner BACK to complete the transfer.
    4.  Say "Goodbye for now."

    SAY THIS EXACTLY: "{script_to_partner}"
    """

    return NodeConfig(
        id="page_24_inform_partner_and_initiate_reconnect_node",
        task_messages=[{"role": "system", "content": task_content}],
        functions=[
        ],
        respond_immediately=True,
        post_actions=[ActionConfig(
            type="function", handler=initiate_consumer_reconnect_after_partner_goodbye)]
    )
