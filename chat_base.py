"""
BASE model demo — llama3.1:8b

The BASE model is a text completer, not a chatbot.
It just predicts the most likely next tokens given your input.
No system prompt, no instruction formatting — raw continuation.

Usage:
  Type any text and press Enter. The model will continue it.
  Commands: /quit  /clear  /help
"""

import sys
import ollama

# ── config ─────────────────────────────────────────────────────────────────
# llama3.1:8b-text-q4_0  is the true base model if you have it pulled.
# llama3.1:8b            is the instruct model — uncomment to compare.
MODEL = "llama3.1:8b-text-q4_0"
# MODEL = "llama3.1:8b"

MAX_TOKENS   = 300   # how many tokens to generate per completion
TEMPERATURE  = 0.8
# ───────────────────────────────────────────────────────────────────────────

TEAL   = "\033[38;5;43m"
AMBER  = "\033[38;5;214m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
CLEAR  = "\033[2J\033[H"


def banner():
    print(f"""
{TEAL}{BOLD}╔══════════════════════════════════════════════╗
║       LLaMA 3.1 8B — BASE MODEL DEMO         ║
╚══════════════════════════════════════════════╝{RESET}

{DIM}Modelo  :{RESET} {AMBER}{MODEL}{RESET}
{DIM}Modo    :{RESET} completación de texto (NO es un asistente)
{DIM}Qué hace:{RESET} predice los tokens más probables dado tu input

{DIM}Escribe cualquier texto y el modelo lo completará.
/quit → salir   /clear → limpiar pantalla   /help → ayuda{RESET}
""")


def help_text():
    print(f"""
{DIM}── Comandos ──────────────────────────────────────{RESET}
  /quit   salir del programa
  /clear  limpiar la pantalla
  /help   mostrar esta ayuda

{DIM}── Consejos para demostrar el BASE model ─────────{RESET}
  • Empezar oraciones a medias:
      "The capital of France is"
      "La mejor manera de cocinar pasta es"

  • Dar contexto conversacional sin instrucción:
      "User: What time is it?\\nAssistant:"
      (el modelo completará como si fuera una conv. cualquiera)

  • Listas o código incompleto:
      "Ingredients for a chocolate cake:\\n1."
      "def fibonacci(n):"
{RESET}""")


def check_model():
    """Verify the model is available locally, offer to pull if not."""
    try:
        local = [m.model for m in ollama.list().models]
    except Exception:
        print(f"\n{AMBER}Error: Ollama no está corriendo.{RESET}")
        print(f"{DIM}Iniciá Ollama con:  ollama serve{RESET}\n")
        sys.exit(1)

    if not any(MODEL in m for m in local):
        print(f"\n{AMBER}Modelo '{MODEL}' no encontrado localmente.{RESET}")
        print(f"Modelos disponibles: {', '.join(local) or 'ninguno'}\n")
        answer = input(f"¿Descargar ahora? ({TEAL}ollama pull {MODEL}{RESET}) [s/N] ").strip().lower()
        if answer in ("s", "si", "sí", "y", "yes"):
            print()
            ollama.pull(MODEL)
        else:
            print(f"\n{DIM}Ejecutá manualmente:  ollama pull {MODEL}{RESET}\n")
            sys.exit(0)


def complete(prompt: str) -> str:
    """Stream a raw text completion and return the full generated text."""
    print(f"\n{DIM}{'─' * 50}{RESET}")
    print(f"{TEAL}", end="", flush=True)

    full = []
    stream = ollama.generate(
        model=MODEL,
        prompt=prompt,
        stream=True,
        options={
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    )
    for chunk in stream:
        token = chunk["response"]
        print(token, end="", flush=True)
        full.append(token)

    print(f"{RESET}\n{DIM}{'─' * 50}{RESET}\n")
    return "".join(full)


def main():
    banner()
    check_model()

    while True:
        try:
            prompt = input(f"{BOLD}Prompt >{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Saliendo...{RESET}")
            break

        if not prompt:
            continue

        match prompt.lower():
            case "/quit" | "/exit":
                print(f"{DIM}Saliendo...{RESET}")
                break
            case "/clear":
                print(CLEAR, end="")
                banner()
            case "/help":
                help_text()
            case _:
                complete(prompt)


if __name__ == "__main__":
    main()
