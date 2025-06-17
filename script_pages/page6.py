# script_pages/page6.py

from typing import Dict
from loguru import logger

from pipecat_flows import ContextStrategy, ContextStrategyConfig, FlowManager, FlowsFunctionSchema, NodeConfig
from .page35 import create_page_35_entry_node
from pipecat.transports.services.daily import DailyTransport
from pipecat_flows.types import ActionConfig


async def end_human_partner_conversation(args: Dict, flow_manager: FlowManager):
    """Transition to the "end_human_partner_conversation" node."""
    await flow_manager.set_node(
        "end_human_partner_conversation", create_end_human_partner_conversation_node()
    )


def create_end_human_partner_conversation_node() -> NodeConfig:
    """Create the "end_human_partner_conversation" node.
    This is the node where the bot tells the partner that they're being patched through to the consumer and ends the conversation (leaving the consumer and partner in the room talking to each other).
    """
    logger.debug("Creating end_human_partner_conversation node.")
    return NodeConfig(
        task_messages=[
            {
                "role": "system",
                "content": "Tell the partner that you're patching them through to the consumer right now.",
            },
        ],
        post_actions=[
            ActionConfig(
                type="function", handler=unmute_consumer_and_make_consumer_partner_hear_each_other),
            ActionConfig(type="end_conversation"),
        ],
        functions=[]
    )


async def unmute_consumer_and_make_consumer_partner_hear_each_other(action: dict, flow_manager: FlowManager):
    """Unmute the consumer and make it so the consumer and human partner can hear each other."""
    transport: DailyTransport = flow_manager.transport
    consumer_participant_id = flow_manager.state.get("consumer_participant_id")
    partner_participant_id = flow_manager.state.get("partner_participant_id")

    await transport.update_remote_participants(
        remote_participants={
            consumer_participant_id: {
                "permissions": {
                    "canSend": ["microphone"],
                    "canReceive": {"byUserId": {"partner": True}},
                },
                "inputsEnabled": {"microphone": True},
            },
            partner_participant_id: {
                "permissions": {"canReceive": {"byUserId": {"consumer": True}}}
            },
        }
    )


async def handle_partner_refusal_transition(args: Dict, flow_manager: FlowManager):
    """Transition to Page 35 if the partner refuses the transfer."""
    logger.info(
        "PAGE 6 TRANSITION: Partner refused transfer. Transitioning to Page 35.")
    await flow_manager.set_node("create_page_35_entry_node", create_page_35_entry_node(flow_manager))


def create_page_6_entry_node(flow_manager: "FlowManager") -> NodeConfig:
    """
    The single, unified node for Page 6.
    It introduces the user to the specialist, listens for the specialist's response,
    and then calls a function to evaluate.
    """
    # Store the user's response in state when it comes in
    flow_manager.state["last_user_message"] = flow_manager.state.get(
        "current_user_message", "")

    partner_name = flow_manager.state.get("bot_name", "Kim")
    lead_first_name = flow_manager.state.get("lead_first_name", "the consumer")

    # Store the greeting line for logging
    greeting = f"Hello specialist, this is {partner_name} from Balboa Digital with {lead_first_name}"
    # flow_manager.state["last_greeting_line"] = greeting

    # Gracefully handle missing debt amount
    debt_amount = flow_manager.state.get("lead_debt_amount", "4000")
    debt_info = f"who is looking for help with ${debt_amount} of unsecured debt"

    introduction_message = f"{greeting} {debt_info}. Are you ready for me to bring them on the line?"
    # Store the greeting for logging
    flow_manager.state["last_greeting_line"] = introduction_message

    logger.debug(
        f"PAGE 6: introduction_message=\'{introduction_message}\'")

    task_content = f"""
    You are now talking to the partner who has just joined the call. Assume the consumer you were speaking with before can no longer hear you.
    Your Job is to speak this greeting to the partner specialist:
    "{introduction_message}"

    Listen carefully to the partner's response.
    - If the partner indicates they are READY to connect to the consumer (e.g., says "yes", "ok", "ready", "bring them on"), call the `connect_human_partner_and_consumer` function.
    - If the partner indicates they REFUSE the transfer or CANNOT take the call (e.g., says "no", "I can't", "unable", "not available"), call the `partner_refused_transfer` function.
    Do not make assumptions. Wait for their explicit response before calling a function.
"""

    return NodeConfig(
        id="page_6_specialist_intro_node",
        task_messages=[{"role": "system", "content": task_content}],
        context_strategy=ContextStrategyConfig(
            strategy=ContextStrategy.RESET,
        ),
        functions=[
            FlowsFunctionSchema(
                name="connect_human_partner_and_consumer",
                description="Call this function if the partner is ready to be connected to the consumer.",
                transition_callback=end_human_partner_conversation,
                properties={},
                required=[],
            ),
            FlowsFunctionSchema(
                name="partner_refused_transfer",
                description="Call this function if the partner refuses the transfer or indicates they cannot take the call.",
                transition_callback=handle_partner_refusal_transition,
                properties={},
                required=[],
            )
        ],
        # Do not allow interruptions during critical partner communication
        interruptions_enabled=False,
        respond_immediately=True
    )
