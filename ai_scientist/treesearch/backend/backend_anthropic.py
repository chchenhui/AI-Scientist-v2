import time
import os

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import anthropic

_client: anthropic.Anthropic = None  # type: ignore
# _client: anthropic.AnthropicBedrock = None  # type: ignore

ANTHROPIC_TIMEOUT_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.APIStatusError,
)


@once
def _setup_anthropic_client():
    global _client
    _client = anthropic.Anthropic(max_retries=0)
    # _client = anthropic.AnthropicBedrock(max_retries=0)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_anthropic_client()

    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 8192  # default for Claude models

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_anthropic_tool_dict]
        filtered_kwargs["tool_choice"] = func_spec.anthropic_tool_choice_dict
        # raise NotImplementedError(
        #     "Anthropic does not support function calling for now."
        # )

    # Anthropic doesn't allow not having a user messages
    # if we only have system msg -> use it as user msg
    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message

    # Anthropic passes the system messages as a separate argument
    if system_message is not None:
        filtered_kwargs["system"] = system_message

    messages = opt_messages_to_list(None, user_message)

    t0 = time.time()
    message = backoff_create(
        _client.messages.create,
        ANTHROPIC_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0
    print(filtered_kwargs)

    if "thinking" in filtered_kwargs:
        assert (
            len(message.content) == 2
            and message.content[0].type == "thinking"
            and message.content[1].type == "text"
        )
        output: str = message.content[1].text
    else:
        assert len(message.content) == 1 and message.content[0].type == "text"
        output: str = message.content[0].text

    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
    }

    return output, req_time, in_tokens, out_tokens, info
