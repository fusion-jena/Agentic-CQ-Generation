import requests
import logging
import time


class LLMNetworkError(RuntimeError):
    """
    Raised by CustomOllamaClient.invoke() when the Ollama server is unreachable
    after all retries (DNS failure, connection refused, or repeated timeouts).

    Callers that can tolerate partial failures (e.g. the per-chunk extractor)
    should catch this explicitly. Callers that cannot (generator, validator,
    refiner) should let it propagate so the pipeline aborts cleanly.
    """


def fmt_duration(seconds: float) -> str:
    """Convert a float seconds value to a human-readable 'Xh Ym Zs Wms' string."""
    seconds = max(0.0, seconds)
    h   = int(seconds // 3600)
    m   = int((seconds % 3600) // 60)
    s   = int(seconds % 60)
    ms  = round((seconds - int(seconds)) * 1000)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m or h:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    parts.append(f"{ms}ms")
    return " ".join(parts)


class CustomOllamaClient:
    def __init__(self, base_url="https://ollama.draco.uni-jena.de/api/generate", model="qwen3:235b-a22b"):
        self.base_url = base_url
        self.model = model
        self._show_url = base_url.replace("/api/generate", "/api/show")
        self._stats_buffer: list = []

    def get_context_length(self) -> int | None:
        """Query Ollama /api/show for the model's actual context window size (in tokens)."""
        try:
            resp = requests.post(self._show_url, json={"name": self.model}, timeout=30)
            resp.raise_for_status()
            model_info = resp.json().get("model_info", {})
            for key, val in model_info.items():
                if "context_length" in key:
                    return int(val)
        except Exception as e:
            logging.warning("[LLM] Could not fetch context length for '%s': %s", self.model, e)
        return None

    def check_connectivity(self, timeout: int = 8) -> bool:
        """
        Quick pre-flight ping: returns True if the Ollama server responds to
        a lightweight /api/tags request within `timeout` seconds.

        Uses /api/tags (GET, no inference) rather than /api/generate so the
        check completes in milliseconds regardless of whether any model is
        loaded into VRAM.  Does NOT raise — callers use the boolean.
        """
        tags_url = self.base_url.replace("/api/generate", "/api/tags")
        try:
            resp = requests.get(tags_url, timeout=timeout)
            return resp.status_code < 500
        except requests.exceptions.RequestException:
            return False

    def invoke(self, prompt: str, format: str | dict | None = None) -> str:
        """Sends a prompt to the Ollama endpoint and returns the text response.

        format: pass "json" to enable Ollama's strict JSON-mode.

        Raises:
            LLMNetworkError: if the server is unreachable after all retries
                (DNS failure, connection refused, or repeated timeouts).
                Callers that tolerate partial failures must catch this explicitly.
        """
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 32768
            }
        }
        if format:
            data["format"] = format

        retries = 5
        for attempt in range(retries):
            try:
                t0 = time.time()
                response = requests.post(self.base_url, json=data, timeout=240)
                response.raise_for_status()
                elapsed = time.time() - t0
                result = response.json()
                prompt_tokens = result.get("prompt_eval_count")
                gen_tokens    = result.get("eval_count")

                logging.info(
                    "[LLM] model='%s' prompt_tokens=%s generated_tokens=%s wall_time_s=%.1f",
                    self.model, prompt_tokens, gen_tokens, elapsed,
                )
                self._stats_buffer.append({
                    "model":            self.model,
                    "wall_time":        fmt_duration(elapsed),
                    "prompt_tokens":    prompt_tokens,
                    "generated_tokens": gen_tokens,
                })
                return result.get("response", "")
            except requests.exceptions.RequestException as e:
                logging.warning(f"LLM API Error (Attempt {attempt+1}/{retries}): {e}")
                time.sleep(2)

        logging.error("LLM API Failed after all retries.")
        raise LLMNetworkError(
            f"Ollama server unreachable after {retries} retries "
            f"(url='{self.base_url}', model='{self.model}'). "
            "Check your VPN / network connection and server status."
        )

    def invoke_with_thinking(self, prompt: str, format: str | None = None) -> tuple[str, str]:
        """
        Like invoke() but with think=True enabled.

        Used for tasks where the chain-of-thought reasoning is itself a
        useful output (e.g. dedup merge_reasoning, cluster justifications).

        Returns:
            (response_text, thinking_text)
            - response_text: the final model output (JSON when format="json")
            - thinking_text: the raw chain-of-thought trace from deepseek-r1.
              Empty string for models that do not emit a thinking field.

        Raises:
            LLMNetworkError: same conditions as invoke().
        """
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": True,
            "options": {
                "temperature": 0.1,
                "num_ctx": 32768,
            },
        }
        if format:
            data["format"] = format

        retries = 5
        for attempt in range(retries):
            try:
                t0 = time.time()
                response = requests.post(self.base_url, json=data, timeout=360)
                response.raise_for_status()
                elapsed = time.time() - t0
                result = response.json()

                response_text = result.get("response", "")
                thinking_text = result.get("thinking", "")   # deepseek-r1 field; empty for others

                prompt_tokens = result.get("prompt_eval_count")
                gen_tokens    = result.get("eval_count")

                thinking_chars = len(thinking_text) if thinking_text else 0

                logging.info(
                    "[LLM] model='%s' think=True prompt_tokens=%s generated_tokens=%s "
                    "thinking_chars=%s wall_time_s=%.1f",
                    self.model, prompt_tokens, gen_tokens, thinking_chars, elapsed,
                )
                self._stats_buffer.append({
                    "model":            self.model,
                    "think":            True,
                    "wall_time":        fmt_duration(elapsed),
                    "prompt_tokens":    prompt_tokens,
                    "generated_tokens": gen_tokens,
                    "thinking_chars":   thinking_chars,
                })
                return response_text, thinking_text

            except requests.exceptions.RequestException as e:
                logging.warning(f"LLM API Error with thinking (Attempt {attempt+1}/{retries}): {e}")
                time.sleep(2)

        logging.error("LLM API Failed after all retries (invoke_with_thinking).")
        raise LLMNetworkError(
            f"Ollama server unreachable after {retries} retries "
            f"(url='{self.base_url}', model='{self.model}'). "
            "Check your VPN / network connection and server status."
        )

    def drain_stats(self) -> list[dict]:
        """Return and clear the accumulated per-call stats since the last drain."""
        stats, self._stats_buffer = self._stats_buffer, []
        return stats
