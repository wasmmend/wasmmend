"""
LLMAgent.py

This module provides an interface to communicate with the OpenAI ChatCompletion API.
Note: please make this decoupled from the repair process. This should be used merely for communication with LLM models.
"""
import os

# API keys are loaded from environment variables. Each *_API_KEYS variable
# may hold a single key or a comma-separated list of keys (rotated on quota
# / rate-limit errors). Set them in your shell before running, e.g.:
#   export OPENAI_API_KEY="sk-..."
#   export DEEPSEEK_API_KEY="sk-...,sk-..."
#   export GEMINI_API_KEY="AIza...,AIza..."
#   export DASHSCOPE_API_KEY="sk-..."
def _load_keys(env_var: str) -> list:
    raw = os.environ.get(env_var, "")
    return [k.strip() for k in raw.split(",") if k.strip()]

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
import json
import time
import logging
import signal
import contextlib
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

from openai import OpenAI


# ---------------------------------------------------------------------------
# Hard timeout for LLM API calls (defends against httpx/urllib3 hangs on
# CLOSE_WAIT sockets where the client's own timeout fails to fire).
# Implemented via SIGALRM — Linux + main thread only; safe inside divergence_guided_repair.py's
# repair loop (which runs single-threaded in each subprocess).
# ---------------------------------------------------------------------------

LLM_HARD_TIMEOUT_SECS = 600  # 5-min wall-clock cap per LLM call


class LLMCallTimeout(Exception):
    """Raised by _hard_timeout when an LLM API call exceeds the wall-clock cap."""
    pass


def _llm_alarm_handler(signum, frame):
    raise LLMCallTimeout("LLM call exceeded hard timeout (signal-based)")


@contextlib.contextmanager
def _hard_timeout(seconds: int):
    """Hard wall-clock timeout via SIGALRM.

    Raises LLMCallTimeout if the wrapped block does not complete within
    `seconds`. Linux + main-thread only.

    When called from a non-main thread (e.g. parallel instrumentation
    ThreadPoolExecutor workers) ``signal.signal`` and
    ``signal.alarm`` raise ``ValueError: signal only works in main
    thread of the main interpreter`` — Python 3.10+ is strict about
    this.  In that case we yield without arming the alarm; the
    SDK-level ``http_options['timeout']`` (300_000 ms) still bounds
    the call.
    """
    import threading
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    prev_handler = signal.signal(signal.SIGALRM, _llm_alarm_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)

# @dataclass
# class ResponseKeys:
#     """
#     ResponseKeys contains the key information extracted from the OpenAI API response.
#     This should include the info like what is the current state, what tool the agent decides to use.
#     """
#     current_state: str = ""
#     using_tool: str = ""


HISTORYS = []

# Global token usage tracker (accumulated across all GeminiAgent instances).
# Kept for backward compatibility with diff_trace_analysis.py which imports this.
GLOBAL_TOKEN_USAGE = {"total_input_tokens": 0, "total_output_tokens": 0}

# Shared error classification used by all agent classes.
NON_RETRYABLE_ERRORS = frozenset({
    "context_too_long", "safety_filter",
})


def classify_api_error(exception: Exception) -> str:
    """Classify an API exception into a human-readable failure reason."""
    err_str = str(exception).lower()
    status = getattr(exception, "status_code", None) or getattr(exception, "code", None)
    if "token count" in err_str and "exceeds" in err_str:
        return "context_too_long"
    if status == 429 or "rate limit" in err_str or "resource exhausted" in err_str:
        return "rate_limit"
    if status == 500 or "internal" in err_str:
        return "server_error"
    if status == 503 or "unavailable" in err_str or "overloaded" in err_str:
        return "service_unavailable"
    if "safety" in err_str or "blocked" in err_str:
        return "safety_filter"
    if "quota" in err_str:
        return "quota_exceeded"
    if "timeout" in err_str or "deadline" in err_str:
        return "timeout"
    if "invalid" in err_str and "api key" in err_str:
        return "auth_error"
    if status == 402 or "insufficient balance" in err_str or "payment" in err_str:
        return "insufficient_balance"
    return "unknown_error"


class ChatGPTAgent:
    """
    ChatGPTAgent provides an interface to communicate with the OpenAI ChatCompletion API.
    It handles configuration, message tracking, API calls, retry logic, and response parsing.
    """

    def __init__(
        self,
        api_key: str = OPENAI_KEY,
        # model: str = "gpt-3.5-turbo",
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        max_tokens: int = 2048,
        max_retries: int = 3,
        backoff_factor: int = 2,
        system_prompt: str = "You are a helpful assistant."
    ):
        """
        Initialize the ChatGPTAgent with the given configurations.

        :param api_key: OpenAI API key
        :param model: The model to use, e.g. 'gpt-3.5-turbo' or 'gpt-4'
        :param temperature: The sampling temperature
        :param max_tokens: The maximum number of tokens in the generated response
        :param max_retries: Maximum number of retry attempts on failure
        :param backoff_factor: The time multiplier to wait between retries (exponential backoff)
        :param system_prompt: Default system prompt to guide the assistant's behavior
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.system_prompt = system_prompt
        self.client = OpenAI(api_key=self.api_key)

        # Configure OpenAI

        # Initialize the conversation with a system message
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Set up basic logging (optional)
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation.
        """
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the conversation.
        """
        self.messages.append({"role": "assistant", "content": content})

    def get_response(self, user_input: str) -> str:
        """
        Send the conversation (including the new user input) to the API,
        handle retries on failures, and return the assistant's response.

        :param user_input: The user's latest message
        :return: The assistant's response as a string
        """
        # Add the new user input to the conversation
        self.add_user_message(user_input)

        # Implement retry logic
        attempt = 0
        while attempt < self.max_retries:
            try:
                self.logger.info(f"Attempt {attempt + 1} to get response.")
                response = self.client.chat.completions.create(model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens)
                # Extract the assistant's message from the response
                assistant_message = response.choices[0].message.content
                # Store it in the conversation
                self.add_assistant_message(assistant_message)
                return assistant_message

            # except openai.error.OpenAIError as e:
            #     self.logger.error(f"OpenAI Error on attempt {attempt + 1}: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")

            attempt += 1
            # Exponential backoff
            sleep_time = self.backoff_factor ** attempt
            self.logger.info(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)

        # If all retries fail, return an error message or raise an exception
        error_msg = "Failed to get response from ChatGPT after multiple retries."
        self.logger.error(error_msg)
        return error_msg

    def reset_conversation(self, system_prompt: Optional[str] = None) -> None:
        """
        Reset the conversation history. Optionally update the system prompt.
        """
        if system_prompt:
            self.system_prompt = system_prompt

        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.logger.info("Conversation has been reset.")


class DeepSeekAgent:
    """
    DeepSeekAgent provides an interface to communicate with the DeepSeek API
    via the OpenAI-compatible endpoint.
    Same interface as GeminiAgent so it can be used as a drop-in replacement.
    """

    AVAILABLE_KEYS = _load_keys("DEEPSEEK_API_KEY")


    def __init__(
        self,
        model: str = "deepseek-chat",
        temperature: float = 0,
        top_p: float = 1,
        max_tokens: int = 8192,
        max_retries: int = 3,
        backoff_factor: int = 2,
        system_prompt: str = "You are a helpful assistant.",
        output_dir: Optional[str] = None,
        log_filename: str = "llm_calls.jsonl",
        **kwargs,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.system_prompt = system_prompt
        self.output_dir = output_dir
        self._log_filename = log_filename

        self._key_index = 0
        self._client = OpenAI(
            api_key=self.AVAILABLE_KEYS[0],
            base_url="https://api.deepseek.com",
            timeout=300.0,
        )

        self.messages: list = []
        self.last_token_usage: dict = {}
        self.total_token_usage: dict = {
            "prompt_tokens": 0,
            "thoughts_tokens": 0,
            "candidates_tokens": 0,
            "total_tokens": 0,
        }
        self._call_number = 0

        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(self.__class__.__name__)

    # ---- public helpers (same signature as GeminiAgent) ----

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def reset_conversation(self, system_prompt: Optional[str] = None) -> None:
        if system_prompt:
            self.system_prompt = system_prompt
        self.messages = []
        self.logger.info("Conversation has been reset.")

    # ---- key rotation ----

    def _rotate_key(self) -> None:
        """Rotate to the next API key in AVAILABLE_KEYS."""
        old_index = self._key_index
        self._key_index = (self._key_index + 1) % len(self.AVAILABLE_KEYS)
        self._client = OpenAI(
            api_key=self.AVAILABLE_KEYS[self._key_index],
            base_url="https://api.deepseek.com",
            timeout=300.0,
        )
        self.logger.info(f"Rotated API key: index {old_index} -> {self._key_index}")

    # ---- logging ----

    def _log_call(self, entry: dict) -> None:
        """Append one JSON line to llm_calls.jsonl in the output directory."""
        if not self.output_dir:
            return
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            log_path = os.path.join(self.output_dir, self._log_filename)
            with open(log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ---- core method ----

    def get_response(self, user_input: str) -> str:
        self.add_user_message(user_input)

        # Keep conversation from growing too large
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]

        # Build messages for the OpenAI-compatible API
        api_messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        for msg in self.messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

        self._call_number += 1
        call_id = self._call_number
        attempts_log = []

        attempt = 0
        while attempt < self.max_retries:
            time.sleep(1 + attempt)
            try:
                self.logger.info(f"Attempt {attempt + 1} to get DeepSeek response.")

                with _hard_timeout(LLM_HARD_TIMEOUT_SECS):
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=api_messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        stream=False,
                    )

                choice = response.choices[0]
                assistant_message = choice.message.content

                if not assistant_message or not assistant_message.strip():
                    attempts_log.append({
                        "attempt": attempt + 1,
                        "status": "retry",
                        "failure_reason": "empty_response",
                    })
                    self.logger.warning(f"Empty response on attempt {attempt + 1}, retrying.")
                    attempt += 1
                    continue

                # Token usage
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                # DeepSeek reasoner may report thinking tokens separately
                thoughts_tokens = 0
                if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                    details = usage.completion_tokens_details
                    thoughts_tokens = getattr(details, "reasoning_tokens", 0) or 0
                candidates_tokens = completion_tokens - thoughts_tokens
                call_total = prompt_tokens + completion_tokens

                self.last_token_usage = {
                    "prompt_tokens": prompt_tokens,
                    "thoughts_tokens": thoughts_tokens,
                    "candidates_tokens": candidates_tokens,
                    "total_tokens": call_total,
                }
                self.total_token_usage["prompt_tokens"] += prompt_tokens
                self.total_token_usage["thoughts_tokens"] += thoughts_tokens
                self.total_token_usage["candidates_tokens"] += candidates_tokens
                self.total_token_usage["total_tokens"] += call_total
                GLOBAL_TOKEN_USAGE["total_input_tokens"] += prompt_tokens
                GLOBAL_TOKEN_USAGE["total_output_tokens"] += completion_tokens

                self.logger.info(
                    f"DeepSeek usage: Input:{prompt_tokens}, "
                    f"Reasoning:{thoughts_tokens}, Output:{candidates_tokens + thoughts_tokens}"
                )

                # Check finish reason for warnings
                warning = None
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason == "length":
                    warning = "max_tokens_exceeded"

                attempts_log.append({
                    "attempt": attempt + 1,
                    "status": "success",
                    "warning": warning,
                    "token_usage": self.last_token_usage,
                })
                self._log_call({
                    "call_number": call_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": self.model,
                    "prompt": user_input,
                    "response": assistant_message,
                    "token_usage": self.last_token_usage,
                    "warning": warning,
                    "attempts": attempts_log,
                })

                self.add_assistant_message(assistant_message)
                return assistant_message

            except Exception as e:
                failure_reason = classify_api_error(e)
                self.logger.error(f"Error on attempt {attempt + 1} [{failure_reason}]: {e}")
                attempts_log.append({
                    "attempt": attempt + 1,
                    "status": "failed" if failure_reason in NON_RETRYABLE_ERRORS else "retry",
                    "failure_reason": failure_reason,
                    "error_message": str(e),
                })
                if failure_reason in NON_RETRYABLE_ERRORS:
                    self.logger.error(f"Non-retryable error ({failure_reason}), stopping immediately.")
                    break
                if failure_reason in ("rate_limit", "quota_exceeded", "auth_error", "insufficient_balance", "timeout"):
                    self._rotate_key()
                    retry_after = getattr(e, "retry_after", self.backoff_factor ** attempt)
                    time.sleep(retry_after)
                else:
                    time.sleep(self.backoff_factor ** attempt)

            attempt += 1

        error_msg = "Failed to get response from DeepSeek after multiple retries."
        self.logger.error(error_msg)
        self._log_call({
            "call_number": call_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "prompt": user_input,
            "response": None,
            "error": error_msg,
            "attempts": attempts_log,
        })
        return error_msg


class GeminiAgent:
    """
    GeminiAgent provides an interface to communicate with the Google Gemini API.
    Same interface as DeepSeekAgent so it can be used as a drop-in replacement.
    """

    AVAILABLE_KEYS = _load_keys("GEMINI_API_KEY")

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        temperature: float = 0,
        top_p: float = 1,
        max_tokens: int = 8192,
        max_retries: int = 3,
        backoff_factor: int = 2,
        system_prompt: str = "You are a helpful assistant.",
        output_dir: Optional[str] = None,
        log_filename: str = "llm_calls.jsonl",
        **kwargs,  # Accept and ignore extra kwargs (e.g., api_key, api_base)
    ):
        from google import genai
        from google.genai import types
        self._genai = genai
        self._types = types

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.system_prompt = system_prompt
        self.output_dir = output_dir
        self._log_filename = log_filename

        self._key_index = 0
        self._client = genai.Client(
            api_key=self.AVAILABLE_KEYS[0],
            http_options={"timeout": 300_000},  # 300s timeout per API call
        )

        # Conversation history (same format as DeepSeekAgent for compatibility).
        # Gemini doesn't use a "system" role in messages — it goes in config.
        self.messages: list = []
        # Per-call token usage (overwritten each successful call)
        self.last_token_usage: dict = {}
        # Cumulative token usage across all successful calls
        self.total_token_usage: dict = {
            "prompt_tokens": 0,
            "thoughts_tokens": 0,
            "candidates_tokens": 0,
            "total_tokens": 0,
        }
        # Call counter for JSONL log
        self._call_number = 0

        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(self.__class__.__name__)

    # ---- public helpers (same signature as DeepSeekAgent) ----

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def reset_conversation(self, system_prompt: Optional[str] = None) -> None:
        if system_prompt:
            self.system_prompt = system_prompt
        self.messages = []
        self.logger.info("Conversation has been reset.")

    # ---- key rotation ----

    def _rotate_key(self) -> None:
        """Rotate to the next API key in AVAILABLE_KEYS."""
        old_index = self._key_index
        self._key_index = (self._key_index + 1) % len(self.AVAILABLE_KEYS)
        self._client = self._genai.Client(
            api_key=self.AVAILABLE_KEYS[self._key_index],
            http_options={"timeout": 300_000},
        )
        self.logger.info(f"Rotated API key: index {old_index} -> {self._key_index}")

    # ---- core method ----

    def _log_call(self, entry: dict) -> None:
        """Append one JSON line to llm_calls.jsonl in the output directory."""
        if not self.output_dir:
            return
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            log_path = os.path.join(self.output_dir, self._log_filename)
            with open(log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass  # Don't fail the repair on log write errors

    # Backward compat aliases — actual logic lives at module level.
    NON_RETRYABLE_ERRORS = NON_RETRYABLE_ERRORS

    @staticmethod
    def _classify_error(exception: Exception) -> str:
        return classify_api_error(exception)

    @staticmethod
    def _check_response_issues(response) -> Optional[str]:
        """Check a successful API response for potential issues.

        Returns a warning string if an issue is detected, else None.
        """
        # Check finish reason (if available)
        if response.candidates:
            candidate = response.candidates[0]
            finish = getattr(candidate, "finish_reason", None)
            # Gemini uses enum values; convert to string for comparison
            finish_str = str(finish).lower() if finish else ""
            if "max_tokens" in finish_str or "length" in finish_str:
                return "max_tokens_exceeded"
            if "safety" in finish_str:
                return "safety_filtered"
            if "recitation" in finish_str:
                return "recitation_blocked"

        # Check for empty/None text
        try:
            text = response.text
        except (ValueError, AttributeError):
            return "empty_response"
        if not text or not text.strip():
            return "empty_response"
        return None

    def get_response(self, user_input: str) -> str:
        self.add_user_message(user_input)

        # Keep conversation from growing too large
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]

        types = self._types

        # Build Gemini contents from message history
        contents = []
        for msg in self.messages:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])],
                )
            )

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            http_options={"timeout": 300_000},
        )

        self._call_number += 1
        call_id = self._call_number
        attempts_log = []  # track all attempts for this call

        attempt = 0
        while attempt < self.max_retries:
            time.sleep(1 + attempt)
            try:
                self.logger.info(f"Attempt {attempt + 1} to get Gemini response.")
                with _hard_timeout(LLM_HARD_TIMEOUT_SECS):
                    response = self._client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config,
                    )

                # Check for response-level issues (truncation, safety, empty)
                issue = self._check_response_issues(response)
                if issue == "empty_response":
                    attempts_log.append({
                        "attempt": attempt + 1,
                        "status": "retry",
                        "failure_reason": "empty_response",
                    })
                    self.logger.warning(f"Empty response on attempt {attempt + 1}, retrying.")
                    attempt += 1
                    continue

                assistant_message = response.text
                # Defensive: max_tokens_exceeded (or any edge case) can still leave
                # response.text as None. Never return None — retry, and if all
                # retries fail the loop falls through to the error_msg path.
                if not assistant_message or not assistant_message.strip():
                    attempts_log.append({
                        "attempt": attempt + 1,
                        "status": "retry",
                        "failure_reason": "empty_or_truncated",
                        "warning_from_check": issue,
                    })
                    self.logger.warning(
                        f"Got None/empty response on attempt {attempt + 1} "
                        f"(check={issue}), retrying."
                    )
                    attempt += 1
                    continue

                usage = response.usage_metadata
                prompt_tokens = usage.prompt_token_count or 0
                thoughts = usage.thoughts_token_count or 0
                candidates = usage.candidates_token_count or 0
                call_total = prompt_tokens + thoughts + candidates
                self.last_token_usage = {
                    "prompt_tokens": prompt_tokens,
                    "thoughts_tokens": thoughts,
                    "candidates_tokens": candidates,
                    "total_tokens": call_total,
                }
                self.total_token_usage["prompt_tokens"] += prompt_tokens
                self.total_token_usage["thoughts_tokens"] += thoughts
                self.total_token_usage["candidates_tokens"] += candidates
                self.total_token_usage["total_tokens"] += call_total
                # Backward-compat: also accumulate into the module-level global
                # so diff_trace_analysis.py can read cross-instance totals.
                GLOBAL_TOKEN_USAGE["total_input_tokens"] += prompt_tokens
                GLOBAL_TOKEN_USAGE["total_output_tokens"] += candidates + thoughts
                self.logger.info(
                    f"Gemini usage: Input:{prompt_tokens}, "
                    f"Reasoning:{thoughts}, Output:{candidates + thoughts}"
                )

                # Log success (with any warnings)
                attempts_log.append({
                    "attempt": attempt + 1,
                    "status": "success",
                    "warning": issue,  # e.g. "max_tokens_exceeded" or None
                    "token_usage": self.last_token_usage,
                })
                self._log_call({
                    "call_number": call_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": self.model,
                    "prompt": user_input,
                    "response": assistant_message,
                    "token_usage": self.last_token_usage,
                    "warning": issue,
                    "attempts": attempts_log,
                })

                self.add_assistant_message(assistant_message)
                return assistant_message

            except Exception as e:
                failure_reason = self._classify_error(e)
                self.logger.error(f"Error on attempt {attempt + 1} [{failure_reason}]: {e}")
                attempts_log.append({
                    "attempt": attempt + 1,
                    "status": "failed" if failure_reason in self.NON_RETRYABLE_ERRORS else "retry",
                    "failure_reason": failure_reason,
                    "error_message": str(e),
                })
                # Don't retry errors that will never succeed with the same input
                if failure_reason in self.NON_RETRYABLE_ERRORS:
                    self.logger.error(f"Non-retryable error ({failure_reason}), stopping immediately.")
                    break
                if failure_reason in ("rate_limit", "quota_exceeded", "auth_error", "insufficient_balance", "timeout"):
                    self._rotate_key()
                    retry_after = getattr(e, "retry_after", self.backoff_factor ** attempt)
                    time.sleep(retry_after)
                else:
                    time.sleep(self.backoff_factor ** attempt)

            attempt += 1

        # All retries exhausted
        error_msg = "Failed to get response from Gemini after multiple retries."
        self.logger.error(error_msg)
        self._log_call({
            "call_number": call_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "prompt": user_input,
            "response": None,
            "error": error_msg,
            "attempts": attempts_log,
        })
        return error_msg


class QwenAgent:
    """
    QwenAgent provides an interface to communicate with Qwen models via
    the DashScope OpenAI-compatible API.
    Same interface as GeminiAgent so it can be used as a drop-in replacement.
    """

    AVAILABLE_KEYS = _load_keys("DASHSCOPE_API_KEY")


    def __init__(
        self,
        model: str = "qwen3.5-plus",
        temperature: float = 0,
        top_p: float = 1,
        max_tokens: int = 8192,
        max_retries: int = 3,
        backoff_factor: int = 2,
        system_prompt: str = "You are a helpful assistant.",
        output_dir: Optional[str] = None,
        log_filename: str = "llm_calls.jsonl",
        enable_thinking: bool = False,
        **kwargs,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.system_prompt = system_prompt
        self.output_dir = output_dir
        self._log_filename = log_filename
        self.enable_thinking = enable_thinking

        self._key_index = 0
        self._client = OpenAI(
            api_key=self.AVAILABLE_KEYS[0],
            base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
            timeout=300.0,
        )

        self.messages: list = []
        self.last_token_usage: dict = {}
        self.total_token_usage: dict = {
            "prompt_tokens": 0,
            "thoughts_tokens": 0,
            "candidates_tokens": 0,
            "total_tokens": 0,
        }
        self._call_number = 0

        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(self.__class__.__name__)

    # ---- public helpers (same signature as GeminiAgent) ----

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def reset_conversation(self, system_prompt: Optional[str] = None) -> None:
        if system_prompt:
            self.system_prompt = system_prompt
        self.messages = []
        self.logger.info("Conversation has been reset.")

    # ---- key rotation ----

    def _rotate_key(self) -> None:
        """Rotate to the next API key in AVAILABLE_KEYS."""
        old_index = self._key_index
        self._key_index = (self._key_index + 1) % len(self.AVAILABLE_KEYS)
        self._client = OpenAI(
            api_key=self.AVAILABLE_KEYS[self._key_index],
            base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
            timeout=300.0,
        )
        self.logger.info(f"Rotated API key: index {old_index} -> {self._key_index}")

    # ---- logging ----

    def _log_call(self, entry: dict) -> None:
        """Append one JSON line to llm_calls.jsonl in the output directory."""
        if not self.output_dir:
            return
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            log_path = os.path.join(self.output_dir, self._log_filename)
            with open(log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ---- core method ----

    def get_response(self, user_input: str) -> str:
        self.add_user_message(user_input)

        # Keep conversation from growing too large
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]

        # Build messages for the OpenAI-compatible API
        api_messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        for msg in self.messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

        self._call_number += 1
        call_id = self._call_number
        attempts_log = []

        attempt = 0
        while attempt < self.max_retries:
            time.sleep(1 + attempt)
            try:
                self.logger.info(f"Attempt {attempt + 1} to get Qwen response.")

                extra_body = {
                    "result_format": "message",
                    "enable_thinking": self.enable_thinking,
                }

                with _hard_timeout(LLM_HARD_TIMEOUT_SECS):
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=api_messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        extra_body=extra_body,
                        stream=False,
                    )

                choice = response.choices[0]
                assistant_message = choice.message.content

                if not assistant_message or not assistant_message.strip():
                    attempts_log.append({
                        "attempt": attempt + 1,
                        "status": "retry",
                        "failure_reason": "empty_response",
                    })
                    self.logger.warning(f"Empty response on attempt {attempt + 1}, retrying.")
                    attempt += 1
                    continue

                # Token usage
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                # Qwen may report thinking tokens separately
                thoughts_tokens = 0
                if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                    details = usage.completion_tokens_details
                    thoughts_tokens = getattr(details, "reasoning_tokens", 0) or 0
                candidates_tokens = completion_tokens - thoughts_tokens
                call_total = prompt_tokens + completion_tokens

                self.last_token_usage = {
                    "prompt_tokens": prompt_tokens,
                    "thoughts_tokens": thoughts_tokens,
                    "candidates_tokens": candidates_tokens,
                    "total_tokens": call_total,
                }
                self.total_token_usage["prompt_tokens"] += prompt_tokens
                self.total_token_usage["thoughts_tokens"] += thoughts_tokens
                self.total_token_usage["candidates_tokens"] += candidates_tokens
                self.total_token_usage["total_tokens"] += call_total
                GLOBAL_TOKEN_USAGE["total_input_tokens"] += prompt_tokens
                GLOBAL_TOKEN_USAGE["total_output_tokens"] += completion_tokens

                self.logger.info(
                    f"Qwen usage: Input:{prompt_tokens}, "
                    f"Reasoning:{thoughts_tokens}, Output:{candidates_tokens + thoughts_tokens}"
                )

                # Check finish reason for warnings
                warning = None
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason == "length":
                    warning = "max_tokens_exceeded"

                attempts_log.append({
                    "attempt": attempt + 1,
                    "status": "success",
                    "warning": warning,
                    "token_usage": self.last_token_usage,
                })
                self._log_call({
                    "call_number": call_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": self.model,
                    "prompt": user_input,
                    "response": assistant_message,
                    "token_usage": self.last_token_usage,
                    "warning": warning,
                    "attempts": attempts_log,
                })

                self.add_assistant_message(assistant_message)
                return assistant_message

            except Exception as e:
                failure_reason = classify_api_error(e)
                self.logger.error(f"Error on attempt {attempt + 1} [{failure_reason}]: {e}")
                attempts_log.append({
                    "attempt": attempt + 1,
                    "status": "failed" if failure_reason in NON_RETRYABLE_ERRORS else "retry",
                    "failure_reason": failure_reason,
                    "error_message": str(e),
                })
                if failure_reason in NON_RETRYABLE_ERRORS:
                    self.logger.error(f"Non-retryable error ({failure_reason}), stopping immediately.")
                    break
                if failure_reason in ("rate_limit", "quota_exceeded", "auth_error", "insufficient_balance", "timeout"):
                    self._rotate_key()
                    retry_after = getattr(e, "retry_after", self.backoff_factor ** attempt)
                    time.sleep(retry_after)
                else:
                    time.sleep(self.backoff_factor ** attempt)

            attempt += 1

        error_msg = "Failed to get response from Qwen after multiple retries."
        self.logger.error(error_msg)
        self._log_call({
            "call_number": call_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "prompt": user_input,
            "response": None,
            "error": error_msg,
            "attempts": attempts_log,
        })
        return error_msg


def create_agent(model: str, **kwargs):
    """Factory: pick the right agent class based on the model name.

    Model name prefixes:
      - "gemini*"    -> GeminiAgent
      - "qwen*"      -> QwenAgent
      - "deepseek*"  -> DeepSeekAgent
      - "gpt*"       -> ChatGPTAgent
    Falls back to GeminiAgent for unknown prefixes.
    """
    m = model.lower()
    if m.startswith("qwen"):
        return QwenAgent(model=model, **kwargs)
    elif m.startswith("deepseek"):
        return DeepSeekAgent(model=model, **kwargs)
    elif m.startswith("gpt"):
        return ChatGPTAgent(model=model, **kwargs)
    else:
        # Default to Gemini (covers "gemini-*" and unknown models)
        return GeminiAgent(model=model, **kwargs)