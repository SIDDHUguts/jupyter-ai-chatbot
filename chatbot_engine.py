"""
Advanced Chatbot Engine Module
Supports:
- Multi-backend LLM execution (Google Gemini 2.5 Flash/Pro & Built-in Smart Offline Engine)
- Streaming response generation
- Notebook Variable Context Injection (DataFrames, arrays, dictionaries)
- Session Save / Load / Fork persistence
- Code block extraction and analysis
"""

from dataclasses import dataclass, field
import datetime
import json
import os
import re
import sys
import time
from typing import List, Optional, Dict, Any, Generator

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv()

PERSONAS: Dict[str, str] = {
    "General Assistant": (
        "You are a helpful, brilliant, and friendly AI assistant. "
        "Provide clear, concise, and structured answers."
    ),
    "Python Coding Expert": (
        "You are a world-class Python engineer. Provide idiomatic, clean, "
        "well-commented Python code solutions with brief explanations."
    ),
    "Data Science & ML Pro": (
        "You are an expert Data Scientist and Machine Learning engineer. "
        "Focus on statistical rigor, data manipulation (pandas, numpy), ML concepts, and best practices."
    ),
    "Friendly Tutor": (
        "You are an encouraging and patient tutor. Break down complex topics into "
        "simple, step-by-step intuitive explanations with analogies."
    ),
    "Creative Storyteller": (
        "You are a creative writer and storyteller with a vivid imagination and expressive style."
    )
}


@dataclass
class ChatMessage:
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%H:%M:%S")
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", ""),
            metadata=data.get("metadata", {})
        )


def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """
    Extract all fenced code blocks from markdown text.
    Returns list of dicts with 'language' and 'code'.
    """
    pattern = r"```([a-zA-Z0-9_\-\+]*)\n(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    blocks = []
    for lang, code in matches:
        blocks.append({
            "language": lang.strip().lower() or "python",
            "code": code.strip()
        })
    return blocks


def format_variable_for_context(name: str, value: Any) -> str:
    """
    Format a Python variable (DataFrame, dict, list, array, etc.)
    into a structured context summary suitable for LLM injection.
    """
    type_name = type(value).__name__

    # Pandas DataFrame inspection
    if type_name == "DataFrame" or hasattr(value, "dtypes"):
        try:
            shape = getattr(value, "shape", ("?", "?"))
            cols = list(getattr(value, "columns", []))
            dtypes_summary = str(getattr(value, "dtypes", ""))[:300]
            head_str = str(value.head(4)) if hasattr(value, "head") else str(value)[:300]
            desc_str = ""
            if hasattr(value, "describe"):
                desc_str = f"\nSummary Stats:\n{str(value.describe())[:400]}"

            return (
                f"### Attached DataFrame `{name}` (Shape: {shape[0]} rows x {shape[1]} cols)\n"
                f"Columns: {cols}\n"
                f"Data Preview (First 4 rows):\n```\n{head_str}\n```{desc_str}\n"
            )
        except Exception:
            return f"### Attached Variable `{name}` (Type: {type_name})\nValue: {str(value)[:400]}\n"

    # NumPy array inspection
    if type_name == "ndarray" or hasattr(value, "ndim"):
        shape = getattr(value, "shape", ("?",))
        dtype = getattr(value, "dtype", "?")
        preview = str(value)[:300]
        return f"### Attached NumPy Array `{name}` (Shape: {shape}, Dtype: {dtype})\nPreview: {preview}\n"

    # Dictionary inspection
    if isinstance(value, dict):
        keys = list(value.keys())[:15]
        preview = json.dumps({k: str(v)[:50] for k, v in list(value.items())[:5]}, indent=2)
        return f"### Attached Dictionary `{name}` ({len(value)} keys: {keys})\nPreview:\n```json\n{preview}\n```\n"

    # General inspection
    val_str = str(value)
    if len(val_str) > 600:
        val_str = val_str[:600] + "... [truncated]"
    return f"### Attached Variable `{name}` (Type: {type_name}):\n```\n{val_str}\n```\n"


class OfflineSmartResponder:
    """
    Advanced offline engine capable of intelligent responses,
    data analysis on attached variables, code generation, debugging, and calculations.
    """

    def generate_response(
        self,
        user_msg: str,
        history: List[ChatMessage],
        persona_prompt: str,
        attached_vars: Dict[str, Any]
    ) -> str:
        text = user_msg.strip()
        lower = text.lower()

        # 1. Attached Variable / DataFrame Analysis
        if attached_vars and any(w in lower for w in ["data", "df", "analyze", "plot", "column", "summary", "shape", "stats"]):
            var_summaries = []
            for name, val in attached_vars.items():
                tname = type(val).__name__
                if tname == "DataFrame" or hasattr(val, "columns"):
                    cols = list(getattr(val, "columns", []))
                    var_summaries.append(
                        f"📊 **Analysis of DataFrame `{name}`:**\n"
                        f"- **Rows:** {val.shape[0]}, **Columns:** {len(cols)} (`{', '.join([str(c) for c in cols[:6]])}`)\n"
                        f"- **Recommended Plot:**\n"
                        f"```python\n"
                        f"import matplotlib.pyplot as plt\n"
                        f"# Quick visualization of {name}\n"
                        f"{name}.plot(kind='bar' if len({name}) < 20 else 'line', figsize=(8, 4))\n"
                        f"plt.title('Overview of {name}')\n"
                        f"plt.grid(True, alpha=0.3)\n"
                        f"plt.show()\n"
                        f"```\n"
                        f"💡 *Click **▶️ Run Code** below to execute this visualization immediately in your notebook!*"
                    )
            if var_summaries:
                return "\n\n".join(var_summaries)

        # 2. Math Calculations
        if any(keyword in lower for keyword in ["calculate", "compute", "solve", "+", "-", "*", "/"]) and re.search(r"\d+\s*[\+\-\*\/]\s*\d+", text):
            try:
                expr = re.sub(r"[^0-9\.\+\-\*\/\(\)\s]", "", text)
                if expr.strip():
                    result = eval(expr, {"__builtins__": None}, {})
                    return f"🧮 **Calculation Result:**\n`{expr.strip()} = {result}`"
            except Exception:
                pass

        # 3. Quick Action: Unit Tests
        if "test" in lower and any(w in lower for w in ["generate", "write", "create", "unit"]):
            return (
                "🧪 **Generated Unit Test Suite:**\n"
                "```python\n"
                "import unittest\n\n"
                "class TestChatbotEngine(unittest.TestCase):\n"
                "    def setUp(self):\n"
                "        # Initialize test fixtures\n"
                "        self.sample_data = [1, 2, 3, 4, 5]\n\n"
                "    def test_computation(self):\n"
                "        result = sum(self.sample_data)\n"
                "        self.assertEqual(result, 15)\n\n"
                "    def test_non_empty(self):\n"
                "        self.assertTrue(len(self.sample_data) > 0)\n\n"
                "# Run tests inline in notebook\n"
                "suite = unittest.TestLoader().loadTestsFromTestCase(TestChatbotEngine)\n"
                "unittest.TextTestRunner(verbosity=2).run(suite)\n"
                "```"
            )

        # 4. Quick Action: Optimization
        if "optimize" in lower or "speed up" in lower or "performance" in lower:
            return (
                "⚡ **Performance Optimization Tips:**\n"
                "1. **Vectorization**: Replace Python `for` loops with NumPy or Pandas vectorized operations.\n"
                "2. **Memory Usage**: Convert `float64` / `int64` to smaller subtypes (`float32`, `int16`, `category`).\n"
                "```python\n"
                "# Optimization Example: Using Vectorization\n"
                "import numpy as np\n"
                "arr = np.arange(1_000_000)\n"
                "# Vectorized fast computation\n"
                "squared = arr ** 2\n"
                "print('Computation complete!')\n"
                "```"
            )

        # 5. Quick Action: Debugging
        if "debug" in lower or "error" in lower or "traceback" in lower:
            return (
                "🐛 **Debugging Assistant:**\n"
                "To debug effectively in Jupyter Notebooks:\n"
                "1. Use `%debug` right after an exception occurs to launch interactive post-mortem debugging.\n"
                "2. Use `print(f'{var=}')` for quick value inspection.\n"
                "3. Wrap risky operations in `try...except Exception as e:` and log `traceback.print_exc()`.\n"
                "```python\n"
                "import traceback\n"
                "try:\n"
                "    # Risky code here\n"
                "    1 / 1\n"
                "    print('No errors encountered!')\n"
                "except Exception as e:\n"
                "    print(f'Caught error: {e}')\n"
                "    traceback.print_exc()\n"
                "```"
            )

        # 6. Plotting / Chart Request
        if "plot" in lower or "chart" in lower or "graph" in lower:
            return (
                "📈 **Interactive Chart Generator:**\n"
                "```python\n"
                "import matplotlib.pyplot as plt\n"
                "import numpy as np\n\n"
                "x = np.linspace(0, 10, 100)\n"
                "y = np.sin(x)\n\n"
                "plt.figure(figsize=(8, 4))\n"
                "plt.plot(x, y, label='sin(x)', color='#4f46e5', linewidth=2)\n"
                "plt.title('Generated Plot Preview', fontsize=14, fontweight='bold')\n"
                "plt.xlabel('X Axis')\n"
                "plt.ylabel('Y Axis')\n"
                "plt.grid(True, linestyle='--', alpha=0.6)\n"
                "plt.legend()\n"
                "plt.tight_layout()\n"
                "plt.show()\n"
                "```"
            )

        # 7. Greetings
        if any(w in lower.split() for w in ["hi", "hello", "hey", "hola", "greetings"]):
            return (
                "👋 **Hello!** I'm your Jupyter AI Chatbot.\n\n"
                "You can:\n"
                "- 📊 Attach notebook variables / DataFrames to chat with them.\n"
                "- ▶️ Run any generated Python code snippet directly with 1 click.\n"
                "- ⚡ Use quick shortcuts (Plot, Debug, Optimize, Test, Explain).\n"
                "- 🔑 Enter your **Gemini API Key** to enable Gemini 2.5 Flash deep reasoning!"
            )

        # 8. General context-grounded response
        turns = len([m for m in history if m.role == 'user'])
        return (
            f"💡 **Response (Offline Mode):**\n\n"
            f"You asked: *\"{user_msg}\"*\n\n"
            f"I have recorded this in conversation history (Turn #{turns}). "
            f"Switch to **Gemini 2.5 Flash** in the model dropdown to unlock full generative AI power!"
        )


class ChatbotEngine:
    """
    Main Chatbot Controller supporting:
    - Multi-backend LLM execution (Gemini 2.5 API & Offline Smart Engine)
    - Streaming responses
    - Variable / DataFrame Context Attachment
    - Session Save / Load / Fork persistence
    - Memory Management
    """

    def __init__(
        self,
        default_persona: str = "General Assistant",
        backend: str = "offline",
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash"
    ):
        self.persona_name = default_persona
        self.custom_system_prompt: Optional[str] = None
        self.backend = backend
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.history: List[ChatMessage] = []
        self.attached_vars: Dict[str, Any] = {}
        self.offline_responder = OfflineSmartResponder()
        self._gemini_client = None

        if self.api_key:
            self._init_gemini()

    def _init_gemini(self) -> bool:
        """Initialize Google GenAI client if API key is present."""
        if not self.api_key:
            return False
        try:
            from google import genai
            self._gemini_client = genai.Client(api_key=self.api_key)
            self.backend = "gemini"
            return True
        except Exception as e:
            print(f"Notice: Could not initialize Gemini client ({e}). Falling back to Offline Mode.")
            self.backend = "offline"
            return False

    def set_backend(self, backend: str, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """Switch backend between 'offline' and 'gemini'."""
        if model_name:
            self.model_name = model_name
        if api_key:
            self.api_key = api_key

        if backend == "gemini":
            if self._init_gemini():
                self.backend = "gemini"
            else:
                self.backend = "offline"
        else:
            self.backend = "offline"

    def set_persona(self, persona_name: str, custom_prompt: Optional[str] = None):
        """Set active persona or custom system instruction."""
        self.persona_name = persona_name
        self.custom_system_prompt = custom_prompt

    def get_system_prompt(self) -> str:
        """Get the effective system instruction including attached variable context."""
        base_prompt = ""
        if self.persona_name == "Custom" and self.custom_system_prompt:
            base_prompt = self.custom_system_prompt
        else:
            base_prompt = PERSONAS.get(self.persona_name, PERSONAS["General Assistant"])

        if not self.attached_vars:
            return base_prompt

        # Inject attached variable summaries into system instruction
        var_context = ["\n\n--- CURRENT NOTEBOOK VARIABLES IN CONTEXT ---"]
        for var_name, var_val in self.attached_vars.items():
            var_context.append(format_variable_for_context(var_name, var_val))
        var_context.append("--- END ATTACHED VARIABLES ---\n")
        return base_prompt + "\n".join(var_context)

    # --- Variable Context Management ---
    def attach_variable(self, name: str, value: Any):
        """Attach a Python variable/dataframe from the notebook to the chat context."""
        self.attached_vars[name] = value

    def detach_variable(self, name: str):
        """Remove a variable from context."""
        if name in self.attached_vars:
            del self.attached_vars[name]

    def clear_attached_variables(self):
        """Clear all attached variables."""
        self.attached_vars.clear()

    # --- Message Generation ---
    def send_message(self, prompt: str) -> str:
        """Send user message, update history, and return assistant response."""
        prompt = prompt.strip()
        if not prompt:
            return ""

        user_msg = ChatMessage(role="user", content=prompt)
        self.history.append(user_msg)

        response_text = ""
        if self.backend == "gemini" and self._gemini_client:
            try:
                system_instruction = self.get_system_prompt()
                contents = []
                for msg in self.history:
                    role = "user" if msg.role == "user" else "model"
                    contents.append(f"{role.capitalize()}: {msg.content}")

                full_prompt = (
                    f"System Instruction:\n{system_instruction}\n\n"
                    + "\n\n".join(contents)
                )

                response = self._gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt
                )
                response_text = response.text or "(No response generated)"
            except Exception as err:
                response_text = (
                    f"⚠️ **Gemini API Error:** {str(err)}\n\n"
                    f"*Falling back to Offline Smart Mode for this turn.*"
                )
                fallback = self.offline_responder.generate_response(
                    prompt, self.history, self.get_system_prompt(), self.attached_vars
                )
                response_text += f"\n\n{fallback}"
        else:
            response_text = self.offline_responder.generate_response(
                prompt, self.history, self.get_system_prompt(), self.attached_vars
            )

        assistant_msg = ChatMessage(role="assistant", content=response_text)
        self.history.append(assistant_msg)
        return response_text

    def send_message_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Yields response tokens/chunks in real time.
        Appends the complete message to history upon completion.
        """
        prompt = prompt.strip()
        if not prompt:
            return

        user_msg = ChatMessage(role="user", content=prompt)
        self.history.append(user_msg)

        accumulated = []

        if self.backend == "gemini" and self._gemini_client:
            try:
                system_instruction = self.get_system_prompt()
                contents = []
                for msg in self.history:
                    role = "user" if msg.role == "user" else "model"
                    contents.append(f"{role.capitalize()}: {msg.content}")

                full_prompt = (
                    f"System Instruction:\n{system_instruction}\n\n"
                    + "\n\n".join(contents)
                )

                stream = self._gemini_client.models.generate_content_stream(
                    model=self.model_name,
                    contents=full_prompt
                )
                for chunk in stream:
                    text_chunk = chunk.text or ""
                    accumulated.append(text_chunk)
                    yield text_chunk
            except Exception as err:
                fallback_header = f"⚠️ **Gemini API Error:** {str(err)}\n\n*Offline response:*\n"
                accumulated.append(fallback_header)
                yield fallback_header
                fallback = self.offline_responder.generate_response(
                    prompt, self.history, self.get_system_prompt(), self.attached_vars
                )
                accumulated.append(fallback)
                yield fallback
        else:
            # Simulated smooth streaming for offline responder
            full_response = self.offline_responder.generate_response(
                prompt, self.history, self.get_system_prompt(), self.attached_vars
            )
            words = full_response.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                accumulated.append(chunk)
                yield chunk
                time.sleep(0.01)

        complete_text = "".join(accumulated)
        assistant_msg = ChatMessage(role="assistant", content=complete_text)
        self.history.append(assistant_msg)

    # --- Session Management ---
    def save_session(self, filepath: str, session_name: Optional[str] = None) -> bool:
        """Save conversation history, settings, and attached variable names to a JSON file."""
        try:
            data = {
                "session_name": session_name or os.path.basename(filepath).replace(".json", ""),
                "saved_at": datetime.datetime.now().isoformat(),
                "persona": self.persona_name,
                "backend": self.backend,
                "model_name": self.model_name,
                "attached_vars": list(self.attached_vars.keys()),
                "history": [msg.to_dict() for msg in self.history]
            }
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False

    def load_session(self, filepath: str) -> bool:
        """Load conversation history and settings from a JSON session file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.persona_name = data.get("persona", "General Assistant")
            self.backend = data.get("backend", "offline")
            self.model_name = data.get("model_name", "gemini-2.5-flash")
            self.history = [ChatMessage.from_dict(m) for m in data.get("history", [])]
            return True
        except Exception as e:
            print(f"Error loading session: {e}")
            return False

    def clear_history(self):
        """Clear all conversation messages."""
        self.history.clear()

    def export_history(self, export_format: str = "markdown") -> str:
        """Export conversation history as Markdown or JSON string."""
        if export_format.lower() == "json":
            return json.dumps([m.to_dict() for m in self.history], indent=2)

        lines = [f"# Chatbot Conversation History ({self.persona_name})\n"]
        lines.append(f"*Exported on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        if self.attached_vars:
            lines.append(f"*Attached Variables in Context:* `{', '.join(self.attached_vars.keys())}`\n")
        lines.append("---\n")
        for msg in self.history:
            role_title = "👤 **User**" if msg.role == "user" else f"🤖 **Assistant ({self.persona_name})**"
            lines.append(f"### {role_title} `[{msg.timestamp}]`\n")
            lines.append(f"{msg.content}\n\n")
        return "".join(lines)
