"""
Unified LLM client supporting OpenAI, Anthropic, and local vLLM backends.

Handles API format differences internally and provides function calling
with a JSON-in-text fallback for models that don't support native tool use.
"""

import base64
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests


class LLMClient:
    """
    Unified LLM client for OpenAI, Anthropic, and vLLM-served models.

    Automatically detects the API format from the api_base URL and handles
    request/response format differences internally.
    """

    def __init__(
        self,
        api_key: str = "EMPTY",
        api_base: str = "http://localhost:8000/v1",
        model: str = "Qwen2.5-VL-7B-Instruct",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: int = 120,
        max_retries: int = 3,
        use_function_calling: Optional[bool] = None,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: API key. Use "EMPTY" for local vLLM servers.
            api_base: Base URL for the API endpoint.
            model: Model name.
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries on failure.
            use_function_calling: Whether to use native function calling.
                None = auto-detect (True for OpenAI API, False for local).
        """
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        # Detect API provider
        self.is_anthropic = "anthropic" in self.api_base.lower()

        # Auto-detect function calling support
        if use_function_calling is None:
            self.use_function_calling = (
                "openai.com" in self.api_base.lower()
                or "azure.com" in self.api_base.lower()
            )
        else:
            self.use_function_calling = use_function_calling

    def chat(
        self,
        system_msg: str,
        user_msg: str,
        images: Optional[List[Union[str, Path]]] = None,
    ) -> str:
        """
        Chat completion with optional vision input.

        Args:
            system_msg: System prompt.
            user_msg: User message.
            images: Optional list of image file paths.

        Returns:
            Response text content.
        """
        if self.is_anthropic:
            return self._chat_anthropic(system_msg, user_msg, images)
        else:
            return self._chat_openai(system_msg, user_msg, images)

    def chat_with_functions(
        self,
        system_msg: str,
        user_msg: str,
        functions: List[Dict],
    ) -> Dict:
        """
        Function calling chat completion.

        For backends supporting native tool use (OpenAI), uses tools/tool_choice.
        For local models, falls back to JSON-in-text parsing.

        Args:
            system_msg: System prompt.
            user_msg: User message.
            functions: List of function schemas in OpenAI tools format.

        Returns:
            Parsed function call arguments as a dict.
        """
        if self.use_function_calling:
            return self._function_call_native(system_msg, user_msg, functions)
        else:
            return self._function_call_fallback(system_msg, user_msg, functions)

    # ---- OpenAI-compatible chat (also used by vLLM) ----

    def _chat_openai(
        self,
        system_msg: str,
        user_msg: str,
        images: Optional[List[Union[str, Path]]] = None,
    ) -> str:
        headers = self._openai_headers()

        # Build user content
        if images:
            user_content = [{"type": "text", "text": user_msg}]
            for img in images:
                b64 = self._encode_image(img)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
        else:
            user_content = user_msg

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
            ],
        }

        endpoint = f"{self.api_base}/chat/completions"
        data = self._request_with_retries(endpoint, headers, payload)
        return data["choices"][0]["message"]["content"]

    # ---- Anthropic chat ----

    def _chat_anthropic(
        self,
        system_msg: str,
        user_msg: str,
        images: Optional[List[Union[str, Path]]] = None,
    ) -> str:
        headers = self._anthropic_headers()

        # Anthropic puts system msg in user content
        if images:
            user_content = [
                {"type": "text", "text": system_msg + "\n\n" + user_msg},
            ]
            for img in images:
                b64 = self._encode_image(img)
                user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                })
        else:
            user_content = [
                {"type": "text", "text": system_msg + "\n\n" + user_msg},
            ]

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": user_content}],
        }

        endpoint = f"{self.api_base}/messages"
        data = self._request_with_retries(endpoint, headers, payload)
        return data["content"][0]["text"]

    # ---- Native function calling (OpenAI) ----

    def _function_call_native(
        self,
        system_msg: str,
        user_msg: str,
        functions: List[Dict],
    ) -> Dict:
        headers = self._openai_headers()

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "tools": functions,
            "tool_choice": "required",
        }

        endpoint = f"{self.api_base}/chat/completions"
        data = self._request_with_retries(endpoint, headers, payload)

        message = data["choices"][0]["message"]

        if "tool_calls" in message and len(message["tool_calls"]) > 0:
            tool_call = message["tool_calls"][0]
            function_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])

            # For planner functions, map function name to tool field
            if function_name == "conclude":
                arguments["tool"] = "conclude"

            return arguments
        else:
            # Fallback: try to parse content as JSON
            if message.get("content"):
                try:
                    return json.loads(message["content"])
                except json.JSONDecodeError:
                    pass
            return {"error": "No function call in response", "tool": "conclude"}

    # ---- Fallback function calling via JSON-in-text ----

    def _function_call_fallback(
        self,
        system_msg: str,
        user_msg: str,
        functions: List[Dict],
    ) -> Dict:
        """Use text-based JSON parsing when native function calling is unavailable."""
        # Build schema description from function definitions
        schema_instructions = self._build_schema_instructions(functions)

        augmented_system = (
            f"{system_msg}\n\n"
            f"## RESPONSE FORMAT INSTRUCTIONS\n"
            f"{schema_instructions}\n\n"
            f"You MUST respond with ONLY a valid JSON object. "
            f"Do NOT include any text before or after the JSON. "
            f"Do NOT wrap it in markdown code blocks."
        )

        response_text = self._chat_openai(augmented_system, user_msg)
        parsed = self._parse_json_from_text(response_text)

        if parsed is None:
            return {"error": "Failed to parse JSON from response", "tool": "conclude"}

        return parsed

    def _build_schema_instructions(self, functions: List[Dict]) -> str:
        """Convert OpenAI function schemas to text instructions for fallback."""
        lines = [
            "You must choose ONE of the following actions and respond "
            "with a JSON object matching its schema:\n"
        ]

        for func_def in functions:
            func = func_def.get("function", func_def)
            name = func["name"]
            desc = func.get("description", "")
            params = func.get("parameters", {})
            props = params.get("properties", {})
            required = params.get("required", [])

            lines.append(f"### Action: {name}")
            lines.append(f"Description: {desc}")
            lines.append("JSON schema:")
            lines.append("{")

            prop_lines = []
            for prop_name, prop_info in props.items():
                prop_type = prop_info.get("type", "string")
                prop_desc = prop_info.get("description", "")
                req_marker = " (REQUIRED)" if prop_name in required else " (optional)"
                enum_values = prop_info.get("enum")
                enum_str = f", one of: {enum_values}" if enum_values else ""
                prop_lines.append(
                    f'  "{prop_name}": <{prop_type}{enum_str}>{req_marker} // {prop_desc}'
                )

            lines.append(",\n".join(prop_lines))
            lines.append("}\n")

        return "\n".join(lines)

    # ---- Helpers ----

    def _openai_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _anthropic_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "anthropic-version": "2023-06-01",
        }

    @staticmethod
    def _encode_image(image_path: Union[str, Path]) -> str:
        """Encode an image file to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _request_with_retries(
        self,
        endpoint: str,
        headers: Dict[str, str],
        payload: Dict,
    ) -> Dict:
        """Make an HTTP POST request with retries."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"API request failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _repair_json_str(s: str) -> str:
        """Best-effort repair of common JSON issues from local models."""
        # Replace single-quoted keys/values with double quotes.
        # Pattern: a single-quoted string used as key or value.
        # This is a heuristic — it handles the most common cases.
        s = re.sub(r"(?<=[{,\s])\s*'([^']+?)'\s*:", r' "\1":', s)   # keys
        s = re.sub(r":\s*'([^']*?)'\s*([,}\]])", r': "\1"\2', s)    # values

        # Remove trailing commas before } or ]
        s = re.sub(r",\s*([}\]])", r"\1", s)

        # Replace Python booleans / None with JSON equivalents
        s = re.sub(r"\bTrue\b", "true", s)
        s = re.sub(r"\bFalse\b", "false", s)
        s = re.sub(r"\bNone\b", "null", s)

        return s

    @staticmethod
    def _try_loads(s: str) -> Optional[Dict]:
        """Try json.loads, then retry after repair."""
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(LLMClient._repair_json_str(s))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _parse_json_from_text(text: str) -> Optional[Dict]:
        """Extract and parse a JSON object from text, handling markdown code blocks."""
        # Try markdown code blocks first
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                result = LLMClient._try_loads(text[start:end].strip())
                if result is not None:
                    return result

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                result = LLMClient._try_loads(text[start:end].strip())
                if result is not None:
                    return result

        # Try to find a JSON object directly
        match = re.search(r"\{", text)
        if match:
            start = match.start()
            # Find matching closing brace
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        result = LLMClient._try_loads(text[start:i + 1])
                        if result is not None:
                            return result
                        break

        # Last resort: try the whole string
        return LLMClient._try_loads(text.strip())
