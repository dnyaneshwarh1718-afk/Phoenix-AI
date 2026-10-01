import asyncio

from rich.console import Console
from rich.panel import Panel

from app.core.config import get_settings
from app.core.logger import configure_logging
from app.llm.gateway import LLMGateway
from app.orchestrator.orchestrator import PhoenixOrchestrator


console = Console()


async def main():
    settings = get_settings()
    configure_logging(settings.log_level)

    phoenix = PhoenixOrchestrator(LLMGateway(settings))

    console.print(Panel.fit("🔥 Phoenix AI — Phase 1", border_style="bright_red"))
    console.print("Type 'exit' to quit.\n")

    while True:
        try:
            message = console.input("[bold cyan]You:[/bold cyan] ")
        except (KeyboardInterrupt, EOFError):
            break

        if message.strip().lower() in {"exit", "quit"}:
            break

        try:
            result = await phoenix.run(message)
            console.print(f"\n[bold green]Phoenix:[/bold green] {result['response']}\n")
            console.print(
                f"[dim]intent={result.get('intent')} | "
                f"agent={result.get('selected_agent')} | "
                f"request_id={result.get('request_id')}[/dim]\n"
            )
        except Exception as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")


if __name__ == "__main__":
    asyncio.run(main())
