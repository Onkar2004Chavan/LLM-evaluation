import os
import requests

class OllamaError(RuntimeError):
    pass

class OllamaClient:
    def __init__(self, base_url: str | None = None, timeout: int = 120):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def list_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            return sorted(item["name"] for item in response.json().get("models", []))
        except requests.RequestException as exc:
            raise OllamaError(f"Cannot connect to Ollama at {self.base_url}. Start Ollama and try again. Details: {exc}") from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaError(f"Ollama returned an invalid model list: {exc}") from exc

    def generate(self, model: str, prompt: str, *, json_mode: bool = False, temperature: float = 0.0) -> str:
        if not model:
            raise OllamaError("No Ollama model selected.")
        try:
            payload = {"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": temperature}}
            if json_mode:
                payload["format"] = "json"
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code == 404:
                raise OllamaError(f"Model '{model}' is not installed in Ollama. Install it manually with: ollama pull {model}")
            response.raise_for_status()
            data = response.json()
            text = data.get("response")
            if not isinstance(text, str) or not text.strip():
                raise OllamaError("Ollama returned an empty response.")
            return text
        except OllamaError:
            raise
        except requests.Timeout as exc:
            raise OllamaError(f"Ollama request timed out after {self.timeout} seconds.") from exc
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama inference failed: {exc}") from exc
        except ValueError as exc:
            raise OllamaError(f"Ollama returned invalid JSON: {exc}") from exc
