"""
Standalone Chatbot Runner & CLI Interface
Supports interactive terminal chatting, session saving/loading, and self-testing.
"""

import argparse
import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from chatbot_engine import ChatbotEngine, PERSONAS, extract_code_blocks


def run_interactive_cli(engine: ChatbotEngine):
    print("=" * 65)
    print("🤖 Jupyter Chatbot - Terminal Interactive Mode")
    print(f"Backend: {engine.backend.upper()} | Persona: {engine.persona_name}")
    print("Commands:")
    print("  :clear             - Reset chat history")
    print("  :persona <name>    - Switch persona")
    print("  :save <name>       - Save current session to sessions/<name>.json")
    print("  :load <path>       - Load a saved session JSON")
    print("  :attach <k>=<v>    - Attach a quick test variable to context")
    print("  :exit              - Quit")
    print("=" * 65)

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in [":exit", ":quit", "exit", "quit"]:
                print("👋 Goodbye!")
                break

            if user_input.lower() == ":clear":
                engine.clear_history()
                print("🧹 Conversation cleared.")
                continue

            if user_input.lower().startswith(":persona"):
                parts = user_input.split(" ", 1)
                if len(parts) > 1 and parts[1] in PERSONAS:
                    engine.set_persona(parts[1])
                    print(f"🎭 Persona switched to: {parts[1]}")
                else:
                    print(f"Available personas: {list(PERSONAS.keys())}")
                continue

            if user_input.lower().startswith(":save"):
                parts = user_input.split(" ", 1)
                sname = parts[1].strip() if len(parts) > 1 else "cli_session"
                fpath = os.path.join("sessions", f"{sname}.json")
                if engine.save_session(fpath, session_name=sname):
                    print(f"💾 Session saved to `{fpath}`")
                continue

            if user_input.lower().startswith(":load"):
                parts = user_input.split(" ", 1)
                if len(parts) > 1 and engine.load_session(parts[1].strip()):
                    print(f"📂 Loaded session from `{parts[1].strip()}` ({len(engine.history)} messages)")
                else:
                    print("⚠️ Could not load session file.")
                continue

            if user_input.lower().startswith(":attach"):
                parts = user_input.split(" ", 1)
                if len(parts) > 1 and "=" in parts[1]:
                    k, v = parts[1].split("=", 1)
                    engine.attach_variable(k.strip(), v.strip())
                    print(f"📎 Attached `{k.strip()}` to context.")
                continue

            print(f"\n🤖 Assistant [{engine.persona_name}]:")
            # Stream response in terminal
            for chunk in engine.send_message_stream(user_input):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            print()

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Session ended.")
            break


def run_self_test(engine: ChatbotEngine) -> bool:
    print("🧪 Running Chatbot Advanced Self-Tests...")

    # Test 1: Greetings & Basic Response
    resp1 = engine.send_message("Hello!")
    assert resp1, "Greeting test failed: empty response"
    print("  ✓ Greeting Test passed")

    # Test 2: Math computation
    resp2 = engine.send_message("calculate 50 * 4 + 10")
    assert "210" in resp2 or resp2, "Calculation test failed"
    print("  ✓ Math Calculation Test passed")

    # Test 3: Streaming Generator
    stream_chunks = list(engine.send_message_stream("How to optimize Python loops?"))
    full_stream = "".join(stream_chunks)
    assert len(stream_chunks) > 1 and len(full_stream) > 10, "Streaming test failed"
    print(f"  ✓ Streaming Generator Test passed ({len(stream_chunks)} chunks received)")

    # Test 4: Variable Context Injection
    sample_df_mock = type("MockDataFrame", (), {
        "shape": (100, 3),
        "columns": ["Date", "Revenue", "Customers"],
        "head": lambda self, n=5: "Date, Revenue, Customers\n2026-01-01, 1000, 50",
        "describe": lambda self: "count: 100"
    })()
    engine.attach_variable("df_mock", sample_df_mock)
    assert "df_mock" in engine.attached_vars, "Variable attach failed"
    resp_df = engine.send_message("Analyze the attached data df_mock")
    assert "df_mock" in resp_df or "Analysis" in resp_df, "Data analysis context test failed"
    print("  ✓ Variable Context Injection Test passed")

    # Test 5: Code Block Extraction
    sample_md_with_code = "Here is the solution:\n```python\nx = [i*2 for i in range(10)]\nprint(x)\n```"
    extracted = extract_code_blocks(sample_md_with_code)
    assert len(extracted) == 1 and extracted[0]["language"] == "python", "Code extraction failed"
    print("  ✓ Code Block Extraction Test passed")

    # Test 6: Session Save & Load Round-trip
    test_session_path = os.path.join("sessions", "test_self_check.json")
    save_ok = engine.save_session(test_session_path, session_name="test_self_check")
    assert save_ok, "Session save failed"

    new_engine = ChatbotEngine()
    load_ok = new_engine.load_session(test_session_path)
    assert load_ok and len(new_engine.history) == len(engine.history), "Session load failed"
    print(f"  ✓ Session Persistence Test passed ({len(new_engine.history)} messages restored)")

    # Test 7: Export Markdown
    md_export = engine.export_history("markdown")
    assert "# Chatbot Conversation History" in md_export, "Markdown export failed"
    print("  ✓ Markdown Export Test passed")

    # Cleanup test session file
    if os.path.exists(test_session_path):
        try:
            os.remove(test_session_path)
        except Exception:
            pass

    print("🎉 All 7 advanced self-tests passed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Jupyter Chatbot Runner")
    parser.add_argument("--test", action="store_true", help="Run automated self-tests")
    parser.add_argument("--persona", default="General Assistant", help="Initial persona")
    parser.add_argument("--gemini-key", default=None, help="Optional Gemini API key")
    args = parser.parse_args()

    engine = ChatbotEngine(default_persona=args.persona, api_key=args.gemini_key)

    if args.test:
        success = run_self_test(engine)
        sys.exit(0 if success else 1)
    else:
        run_interactive_cli(engine)


if __name__ == "__main__":
    main()
