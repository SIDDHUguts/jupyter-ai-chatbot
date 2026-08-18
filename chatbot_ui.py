"""
Advanced Jupyter Chatbot UI Module
Renders an interactive, feature-packed chat interface inside Jupyter Notebook cells:
- Multi-Theme Engine (Midnight Dark, Clean Light, Cyberpunk Glass)
- Quick Action Shortcuts (Plot, Debug, Optimize, Test, Analyze)
- One-Click Code Execution in Jupyter Kernel
- Variable Inspector & DataFrame Context Injector
- Session Save & Load Manager
"""

import html
import io
import json
import os
import re
import sys
import traceback
from typing import Optional, Dict, Any, List
import ipywidgets as widgets
from IPython.display import display, HTML
from chatbot_engine import ChatbotEngine, PERSONAS, extract_code_blocks

THEMES = {
    "Midnight Dark": {
        "bg_main": "#0f172a",
        "bg_chat": "#1e293b",
        "bg_toolbar": "#1e293b",
        "bg_user_bubble": "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
        "text_user": "#ffffff",
        "bg_bot_bubble": "#334155",
        "text_bot": "#f8fafc",
        "border": "#334155",
        "code_bg": "#0b0f19",
        "header_grad": "linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)",
        "accent": "#818cf8"
    },
    "Clean Light": {
        "bg_main": "#ffffff",
        "bg_chat": "#ffffff",
        "bg_toolbar": "#f8fafc",
        "bg_user_bubble": "linear-gradient(135deg, #2563eb 0%, #3b82f6 100%)",
        "text_user": "#ffffff",
        "bg_bot_bubble": "#f1f5f9",
        "text_bot": "#1e293b",
        "border": "#e2e8f0",
        "code_bg": "#1e293b",
        "header_grad": "linear-gradient(135deg, #2563eb 0%, #4f46e5 100%)",
        "accent": "#2563eb"
    },
    "Cyberpunk Glass": {
        "bg_main": "#090d16",
        "bg_chat": "#0d1322",
        "bg_toolbar": "#111827",
        "bg_user_bubble": "linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%)",
        "text_user": "#ffffff",
        "bg_bot_bubble": "#1e293b",
        "text_bot": "#38bdf8",
        "border": "#06b6d4",
        "code_bg": "#030712",
        "header_grad": "linear-gradient(135deg, #06b6d4 0%, #9333ea 100%)",
        "accent": "#06b6d4"
    }
}


def render_markdown(text: str, code_bg: str = "#1e1e2e") -> str:
    """Format markdown text with code highlighting, bold, italics, tables, and lists."""
    escaped = html.escape(text)

    # Multi-line Code blocks
    def code_sub(match):
        code = match.group(2)
        return (
            f'<pre style="background: {code_bg}; color: #cdd6f4; padding: 12px; '
            f'border-radius: 8px; overflow-x: auto; font-family: Consolas, Monaco, monospace; '
            f'font-size: 12.5px; margin: 8px 0; border: 1px solid rgba(255,255,255,0.1);">'
            f'<code>{code}</code></pre>'
        )

    escaped = re.sub(r"```([a-zA-Z0-9_\-\+]*)\n(.*?)```", code_sub, escaped, flags=re.DOTALL)

    # Inline code `code`
    escaped = re.sub(
        r"`([^`]+)`",
        r'<code style="background: rgba(120, 120, 140, 0.2); color: #f43f5e; padding: 2px 5px; border-radius: 4px; font-family: monospace; font-size: 12px;">\1</code>',
        escaped
    )

    # Bold **text**
    escaped = re.sub(r"\*\*([^\*]+)\*\*", r'<strong>\1</strong>', escaped)

    # Italic *text*
    escaped = re.sub(r"\*([^\*]+)\*", r'<em>\1</em>', escaped)

    # Line breaks
    lines = escaped.split('\n')
    out_lines = []
    in_pre = False
    for line in lines:
        if '<pre' in line:
            in_pre = True
        if '</pre>' in line:
            in_pre = False
            out_lines.append(line)
            continue
        if in_pre:
            out_lines.append(line)
        else:
            out_lines.append(line + '<br/>')

    return '\n'.join(out_lines)


class JupyterChatUI:
    """
    Advanced interactive Jupyter Chatbot widget.
    """

    def __init__(self, engine: Optional[ChatbotEngine] = None, user_namespace: Optional[Dict[str, Any]] = None):
        self.engine = engine or ChatbotEngine()
        self.user_namespace = user_namespace or {}
        self.theme_name = "Midnight Dark"
        self.sessions_dir = "sessions"
        os.makedirs(self.sessions_dir, exist_ok=True)
        self._build_ui()

    def _build_ui(self):
        theme = THEMES[self.theme_name]

        # 1. Header Widget
        self.header_html = widgets.HTML(
            value=self._render_header_html()
        )

        # 2. Controls Toolbar - Row 1 (Persona, Model, Theme, Clear)
        self.persona_dropdown = widgets.Dropdown(
            options=list(PERSONAS.keys()) + ["Custom"],
            value=self.engine.persona_name,
            description="Persona:",
            layout=widgets.Layout(width="210px")
        )
        self.persona_dropdown.observe(self._on_persona_change, names="value")

        self.backend_dropdown = widgets.Dropdown(
            options=["Offline Smart Mode", "Gemini 2.5 Flash", "Gemini 1.5 Pro"],
            value="Offline Smart Mode" if self.engine.backend == "offline" else "Gemini 2.5 Flash",
            description="Model:",
            layout=widgets.Layout(width="190px")
        )
        self.backend_dropdown.observe(self._on_backend_change, names="value")

        self.theme_dropdown = widgets.Dropdown(
            options=list(THEMES.keys()),
            value=self.theme_name,
            description="Theme:",
            layout=widgets.Layout(width="190px")
        )
        self.theme_dropdown.observe(self._on_theme_change, names="value")

        self.btn_clear = widgets.Button(
            description="Clear",
            icon="trash",
            button_style="warning",
            tooltip="Clear Chat History",
            layout=widgets.Layout(width="75px")
        )
        self.btn_clear.on_click(self._on_clear_click)

        self.toolbar_row1 = widgets.HBox(
            [self.persona_dropdown, self.backend_dropdown, self.theme_dropdown, self.btn_clear],
            layout=widgets.Layout(padding="6px 10px", background=theme["bg_toolbar"], align_items="center")
        )

        # 3. Controls Toolbar - Row 2 (API Key, Session Save/Load, Export)
        self.api_key_input = widgets.Password(
            placeholder="Gemini API Key (optional)",
            description="API Key:",
            layout=widgets.Layout(width="220px")
        )
        self.btn_connect_api = widgets.Button(
            description="Connect",
            button_style="primary",
            tooltip="Connect Gemini API Key",
            layout=widgets.Layout(width="80px")
        )
        self.btn_connect_api.on_click(self._on_connect_api)

        self.session_name_input = widgets.Text(
            placeholder="session_name",
            description="Session:",
            layout=widgets.Layout(width="180px")
        )

        self.btn_save_session = widgets.Button(
            description="Save",
            icon="save",
            button_style="info",
            tooltip="Save current conversation session",
            layout=widgets.Layout(width="75px")
        )
        self.btn_save_session.on_click(self._on_save_session)

        self.btn_export = widgets.Button(
            description="Export",
            icon="download",
            button_style="",
            tooltip="Export Conversation Markdown",
            layout=widgets.Layout(width="80px")
        )
        self.btn_export.on_click(self._on_export_click)

        self.toolbar_row2 = widgets.HBox(
            [self.api_key_input, self.btn_connect_api, self.session_name_input, self.btn_save_session, self.btn_export],
            layout=widgets.Layout(padding="0 10px 8px 10px", background=theme["bg_toolbar"], align_items="center")
        )

        # 4. Quick Actions Toolbar (Prompt Shortcuts)
        self.btn_qa_analyze = widgets.Button(description="📊 Analyze Data", layout=widgets.Layout(width="115px"))
        self.btn_qa_plot = widgets.Button(description="📈 Plot Chart", layout=widgets.Layout(width="105px"))
        self.btn_qa_debug = widgets.Button(description="🐛 Debug Code", layout=widgets.Layout(width="110px"))
        self.btn_qa_optimize = widgets.Button(description="⚡ Optimize", layout=widgets.Layout(width="100px"))
        self.btn_qa_test = widgets.Button(description="🧪 Unit Tests", layout=widgets.Layout(width="105px"))

        self.btn_qa_analyze.on_click(lambda b: self._inject_quick_action("Analyze the current dataset / variables in context and summarize key statistics."))
        self.btn_qa_plot.on_click(lambda b: self._inject_quick_action("Generate a high-quality visualization plot using matplotlib/seaborn."))
        self.btn_qa_debug.on_click(lambda b: self._inject_quick_action("Review this code for bugs, errors, and performance bottlenecks."))
        self.btn_qa_optimize.on_click(lambda b: self._inject_quick_action("How can I optimize this code for maximum execution speed and lower memory usage?"))
        self.btn_qa_test.on_click(lambda b: self._inject_quick_action("Write a comprehensive unit test suite with edge cases."))

        self.quick_actions_bar = widgets.HBox(
            [widgets.Label("⚡ Shortcuts:"), self.btn_qa_analyze, self.btn_qa_plot, self.btn_qa_debug, self.btn_qa_optimize, self.btn_qa_test],
            layout=widgets.Layout(padding="4px 10px", background=theme["bg_toolbar"], align_items="center", border_top=f"1px solid {theme['border']}")
        )

        # 5. Variable Context Injector Bar
        self.var_name_input = widgets.Text(
            placeholder="e.g. df_sales, data_dict",
            description="Attach Var:",
            layout=widgets.Layout(width="240px")
        )
        self.btn_attach_var = widgets.Button(
            description="Attach to Context",
            icon="paperclip",
            button_style="success",
            layout=widgets.Layout(width="150px")
        )
        self.btn_attach_var.on_click(self._on_attach_var_click)

        self.attached_vars_label = widgets.HTML(
            value=self._render_attached_vars_html(),
            layout=widgets.Layout(flex="1 1 auto", padding="0 8px")
        )

        self.var_bar = widgets.HBox(
            [self.var_name_input, self.btn_attach_var, self.attached_vars_label],
            layout=widgets.Layout(padding="4px 10px", background=theme["bg_toolbar"], align_items="center")
        )

        # 6. Chat Display Area
        self.chat_display = widgets.HTML(
            value=self._render_chat_html(),
            layout=widgets.Layout(
                height="400px",
                overflow="y-auto",
                border=f"1px solid {theme['border']}",
                padding="14px",
                background=theme["bg_chat"]
            )
        )

        # 7. Code Execution Widget Area (Interactive In-Kernel Runner)
        self.code_runner_output = widgets.Output(
            layout=widgets.Layout(
                max_height="250px",
                overflow="y-auto",
                border="1px dashed #6366f1",
                padding="8px",
                margin="6px 0",
                display="none"
            )
        )

        # 8. Input & Send Box
        self.input_text = widgets.Text(
            placeholder="Ask anything, request code, or chat with attached data... (Press Enter or click Send)",
            layout=widgets.Layout(flex="1 1 auto")
        )
        self.input_text.on_submit(self._on_send_click)

        self.btn_send = widgets.Button(
            description="Send",
            icon="paper-plane",
            button_style="success",
            layout=widgets.Layout(width="90px")
        )
        self.btn_send.on_click(self._on_send_click)

        self.btn_run_last_code = widgets.Button(
            description="▶️ Run Last Code",
            icon="play",
            button_style="primary",
            tooltip="Execute the latest Python snippet generated by assistant in notebook kernel",
            layout=widgets.Layout(width="145px")
        )
        self.btn_run_last_code.on_click(self._on_run_latest_code)

        self.input_row = widgets.HBox(
            [self.input_text, self.btn_send, self.btn_run_last_code],
            layout=widgets.Layout(padding="10px", background=theme["bg_toolbar"], border_radius="0 0 14px 14px")
        )

        # 9. Feedback & Status Output
        self.status_output = widgets.Output()

        # Combine into Main Layout
        self.main_container = widgets.VBox(
            [
                self.header_html,
                self.toolbar_row1,
                self.toolbar_row2,
                self.var_bar,
                self.quick_actions_bar,
                self.chat_display,
                self.code_runner_output,
                self.input_row,
                self.status_output
            ],
            layout=widgets.Layout(
                width="100%",
                max_width="920px",
                border=f"1px solid {theme['border']}",
                border_radius="14px",
                box_shadow="0 10px 30px rgba(0, 0, 0, 0.25)",
                margin="15px 0",
                background=theme["bg_main"]
            )
        )

    def _render_header_html(self) -> str:
        theme = THEMES[self.theme_name]
        backend_badge = "⚡ Offline Mode" if self.engine.backend == "offline" else f"✨ {self.engine.model_name}"
        return f"""
        <div style="background: {theme['header_grad']};
                    color: white; padding: 12px 18px; border-radius: 12px 12px 0 0;
                    display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 26px;">🤖</span>
                <div>
                    <div style="font-weight: 700; font-size: 16px; letter-spacing: 0.3px;">Jupyter AI Assistant & Code Companion</div>
                    <div style="font-size: 11px; opacity: 0.9;">Multi-Backend • In-Kernel Execution • Data Context Inspector</div>
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.22); padding: 4px 12px;
                        border-radius: 20px; font-size: 12px; font-weight: 600;">
                {backend_badge}
            </div>
        </div>
        """

    def _render_attached_vars_html(self) -> str:
        if not self.engine.attached_vars:
            return '<span style="font-size: 11.5px; opacity: 0.7;">No variables attached. Type a variable name above to attach DataFrame/Data.</span>'
        chips = []
        for name in self.engine.attached_vars.keys():
            chips.append(f'<span style="background: #4f46e5; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; margin-right: 4px;">📎 {name}</span>')
        return "".join(chips)

    def _render_chat_html(self) -> str:
        theme = THEMES[self.theme_name]
        if not self.engine.history:
            return f"""
            <div style="text-align: center; color: #94a3b8; padding: 50px 10px; font-family: sans-serif;">
                <div style="font-size: 36px; margin-bottom: 8px;">💬</div>
                <div style="font-size: 16px; font-weight: 600; color: {theme['accent']};">Jupyter Chatbot is ready</div>
                <div style="font-size: 12.5px; margin-top: 6px; opacity: 0.8;">
                    Use shortcuts above, attach a DataFrame, or ask any question to begin!
                </div>
            </div>
            """

        html_blocks = ['<div style="display: flex; flex-direction: column; gap: 14px; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;">']
        for msg in self.engine.history:
            is_user = (msg.role == "user")
            align = "flex-end" if is_user else "flex-start"
            bubble_bg = theme["bg_user_bubble"] if is_user else theme["bg_bot_bubble"]
            text_color = theme["text_user"] if is_user else theme["text_bot"]
            border_radius = "16px 16px 2px 16px" if is_user else "16px 16px 16px 2px"
            avatar = "👤" if is_user else "🤖"
            role_label = "You" if is_user else self.engine.persona_name
            shadow = "0 3px 10px rgba(0,0,0,0.18)"

            content_html = render_markdown(msg.content, code_bg=theme["code_bg"])

            block = f"""
            <div style="display: flex; justify-content: {align}; width: 100%;">
                <div style="max-width: 85%; background: {bubble_bg}; color: {text_color};
                            padding: 11px 16px; border-radius: {border_radius};
                            box-shadow: {shadow}; font-size: 13.5px; line-height: 1.5; word-break: break-word;">
                    <div style="display: flex; justify-content: space-between; align-items: center;
                                margin-bottom: 5px; font-size: 11px; opacity: 0.8; gap: 14px;">
                        <span><strong>{avatar} {role_label}</strong></span>
                        <span>{msg.timestamp}</span>
                    </div>
                    <div>{content_html}</div>
                </div>
            </div>
            """
            html_blocks.append(block)

        html_blocks.append('</div>')
        return "\n".join(html_blocks)

    def _update_view(self):
        self.chat_display.value = self._render_chat_html()
        self.attached_vars_label.value = self._render_attached_vars_html()
        self.header_html.value = self._render_header_html()

    def _on_send_click(self, sender):
        text = self.input_text.value.strip()
        if not text:
            return

        self.input_text.value = ""
        self.btn_send.disabled = True
        self.input_text.disabled = True

        with self.status_output:
            self.status_output.clear_output()

        self.engine.send_message(text)
        self._update_view()

        self.btn_send.disabled = False
        self.input_text.disabled = False

    def _inject_quick_action(self, prompt: str):
        self.input_text.value = prompt
        self._on_send_click(None)

    def _on_attach_var_click(self, btn):
        var_name = self.var_name_input.value.strip()
        if not var_name:
            return

        # Look in user_namespace or __main__ globals
        val = None
        if var_name in self.user_namespace:
            val = self.user_namespace[var_name]
        elif "__main__" in sys.modules and hasattr(sys.modules["__main__"], var_name):
            val = getattr(sys.modules["__main__"], var_name)
        else:
            # Check builtins / globals
            import __main__
            val = getattr(__main__, var_name, None)

        with self.status_output:
            self.status_output.clear_output()
            if val is not None:
                self.engine.attach_variable(var_name, val)
                self.var_name_input.value = ""
                self._update_view()
                print(f"✅ Attached variable `{var_name}` ({type(val).__name__}) to chat context!")
            else:
                print(f"⚠️ Variable `{var_name}` not found in the current Python namespace.")

    def _on_run_latest_code(self, btn):
        """Extract latest code block from assistant history and execute in active kernel."""
        with self.status_output:
            self.status_output.clear_output()

        # Find latest assistant message with code
        latest_code = ""
        for msg in reversed(self.engine.history):
            if msg.role == "assistant":
                blocks = extract_code_blocks(msg.content)
                if blocks:
                    latest_code = blocks[0]["code"]
                    break

        if not latest_code:
            with self.status_output:
                print("ℹ️ No code blocks found in recent assistant messages.")
            return

        self.code_runner_output.layout.display = "block"
        with self.code_runner_output:
            print(f"▶️ Executing code in Jupyter Kernel...\n" + "-" * 40)
            try:
                # Execute in __main__ namespace so variables persist in user notebook
                import __main__
                exec_globals = self.user_namespace if self.user_namespace else __main__.__dict__
                exec(latest_code, exec_globals)
                print("\n✅ Code execution finished successfully!")
            except Exception as e:
                print(f"\n❌ Execution Error:\n{traceback.format_exc()}")

    def _on_persona_change(self, change):
        self.engine.set_persona(change["new"])
        with self.status_output:
            self.status_output.clear_output()
            print(f"🎭 Switched persona to: {change['new']}")
        self._update_view()

    def _on_backend_change(self, change):
        choice = change["new"]
        if choice == "Offline Smart Mode":
            self.engine.set_backend("offline")
        elif "Gemini" in choice:
            model = "gemini-2.5-flash" if "2.5" in choice else "gemini-1.5-pro"
            self.engine.set_backend("gemini", model_name=model)
        self._update_view()

    def _on_theme_change(self, change):
        self.theme_name = change["new"]
        theme = THEMES[self.theme_name]
        self.main_container.layout.background = theme["bg_main"]
        self.main_container.layout.border = f"1px solid {theme['border']}"
        self.chat_display.layout.background = theme["bg_chat"]
        self.chat_display.layout.border = f"1px solid {theme['border']}"
        self.toolbar_row1.layout.background = theme["bg_toolbar"]
        self.toolbar_row2.layout.background = theme["bg_toolbar"]
        self.var_bar.layout.background = theme["bg_toolbar"]
        self.quick_actions_bar.layout.background = theme["bg_toolbar"]
        self.input_row.layout.background = theme["bg_toolbar"]
        self._update_view()

    def _on_connect_api(self, btn):
        key = self.api_key_input.value.strip()
        with self.status_output:
            self.status_output.clear_output()
            if not key:
                print("⚠️ Please enter a valid Gemini API Key.")
                return
            self.engine.set_backend("gemini", api_key=key)
            if self.engine.backend == "gemini":
                print("✅ Connected to Google Gemini API!")
                self._update_view()
            else:
                print("❌ Could not connect to Gemini API. Please check your key.")

    def _on_save_session(self, btn):
        name = self.session_name_input.value.strip() or f"session_{int(io.StringIO().tell() or 1)}"
        filepath = os.path.join(self.sessions_dir, f"{name}.json")
        success = self.engine.save_session(filepath, session_name=name)
        with self.status_output:
            self.status_output.clear_output()
            if success:
                print(f"💾 Session saved to `{filepath}`!")
            else:
                print(f"❌ Failed to save session.")

    def _on_clear_click(self, btn):
        self.engine.clear_history()
        self.code_runner_output.layout.display = "none"
        self._update_view()
        with self.status_output:
            self.status_output.clear_output()
            print("🧹 Chat history cleared.")

    def _on_export_click(self, btn):
        md = self.engine.export_history("markdown")
        with self.status_output:
            self.status_output.clear_output()
            print("📁 Markdown Transcript:\n\n" + md)

    def show(self):
        """Render widget in Jupyter cell."""
        display(self.main_container)
        return self.main_container
