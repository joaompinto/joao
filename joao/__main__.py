#!/usr/bin/env python3
import sys
import argparse
import os
from rich.console import Console
from rich.markdown import Markdown
from .agent import Agent

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_response(response: str, raw: bool = False, console: Console = None):
    """Print a complete response without streaming"""
    if raw:
        print(response)
    else:
        md = Markdown(response)
        console.print(md)

def process_stream(stream, raw=False, console=None):
    """Process a stream of text, buffering until we have complete lines"""
    buffer = ""
    for chunk in stream:
        if chunk:
            if raw:
                print(chunk, end='')
                sys.stdout.flush()
            else:
                buffer += chunk
                lines = buffer.splitlines(True)  # Keep line endings
                if len(lines) > 1:  # We have complete lines
                    for line in lines[:-1]:  # Process all complete lines
                        if line.strip():  # Skip empty lines
                            md = Markdown(line)
                            console.print(md)
                    buffer = lines[-1]  # Keep partial line in buffer
    
    # Process any remaining text
    if buffer and not raw:
        md = Markdown(buffer)
        console.print(md)

def main():
    parser = argparse.ArgumentParser(description="Simple OpenAI chat client")
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt to send to the model. If not provided, enters chat mode. Type 'reset' to reset conversation."
    )
    parser.add_argument(
        "-s", "--system",
        help="System prompt to use",
        default="You are a helpful assistant."
    )
    parser.add_argument(
        "-e", "--env",
        help="Environment prefix for variables (e.g., 'ALIBABA' for ALIBABA_OPENAI_API_KEY)",
        default=None
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0,
        help="Sampling temperature (0.0-2.0), higher is more random"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug info about style markers"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Show raw output without markdown formatting"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response token by token"
    )
    
    args = parser.parse_args()
    
    if not 0 <= args.temperature <= 2:
        print("Error: Temperature must be between 0 and 2", file=sys.stderr)
        sys.exit(1)
    
    console = Console()
    
    # Convert environment prefix to uppercase if provided
    env_prefix = args.env.upper() if args.env else None
    agent = Agent(args.system, temperature=args.temperature, tenant_prefix=env_prefix)
    
    if args.prompt is None:  # Chat mode
        def print_config(agent):
            console.print("\nModel: ", style="blue", end='')
            console.print(agent.model, style="green")
            console.print("Temperature: ", style="blue", end='')
            console.print(f"{agent.temperature}", style="green", end='')
            console.print(" [dim]⟨0.0-2.0⟩[/dim]", style="blue")
            console.print("System: ", style="blue", end='')
            console.print(agent.system_prompt or 'No system prompt set', style="green")
        
        def reset_conversation(new_system=None):
            clear_screen()
            return Agent(new_system or args.system, temperature=args.temperature, tenant_prefix=env_prefix)
        
        print_config(agent)
        console.print("\nStarting chat session. Commands:")
        console.print("  /reset             - Clear conversation")
        console.print("  /reset <prompt>    - Clear conversation and set new system prompt")
        console.print("  Ctrl+C            - Exit session")
        
        while True:
            try:
                console.print("\nYou: ", style="green", end='')
                user_input = input("").strip()
                    
                if not user_input:  # Just whitespace
                    continue
                
                if user_input.startswith("/reset"):
                    # Extract new system prompt if provided
                    new_system = user_input[6:].strip() if len(user_input) > 6 else None
                    agent = reset_conversation(new_system)
                    print_config(agent)
                    if new_system:
                        console.print("\nConversation reset with new system prompt.")
                    else:
                        console.print("\nConversation reset.")
                    continue
                
                console.print("\nAssistant:", style="blue")
                
                response = agent.request(user_input, stream=args.stream)
                if args.stream:
                    process_stream(response, args.raw, console)
                else:
                    print_response(response, args.raw, console)
            
            except KeyboardInterrupt:
                console.print("\nExiting chat session...")
                break
    else:
        # Single prompt mode
        response = agent.request(args.prompt, stream=args.stream)
        if args.stream:
            process_stream(response, args.raw, console)
        else:
            print_response(response, args.raw, console)

if __name__ == "__main__":
    main()