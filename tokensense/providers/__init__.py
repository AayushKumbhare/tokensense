from . import anthropic, gemini, ollama, openai

REGISTRY = {
    anthropic.PROVIDER_NAME: anthropic,
    openai.PROVIDER_NAME: openai,
    gemini.PROVIDER_NAME: gemini,
    ollama.PROVIDER_NAME: ollama,
}


def get_provider(name: str):
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown provider '{name}'. Choose from: {', '.join(REGISTRY)}") from exc
