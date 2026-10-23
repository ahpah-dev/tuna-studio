"""Context-bound response budgets and model-native reasoning text routing."""
from __future__ import annotations


def response_budget(requested, prompt_tokens, context):
    remaining = context - prompt_tokens
    if remaining < 1 or (requested is not None and requested > remaining):
        raise ValueError(f"This conversation exceeds the {context:,}-token chat budget. Start a new chat or shorten your message; the saved policy will not be truncated.")
    return remaining if requested is None else requested


def reasoning_supported(tokenizer, messages):
    options = {"tokenize":False,"add_generation_prompt":True}
    return tokenizer.apply_chat_template(messages,enable_thinking=True,**options) != tokenizer.apply_chat_template(messages,enable_thinking=False,**options)


def finish_reason(cancelled, count, budget, unlimited, last_token, eos):
    if cancelled:
        return "cancelled"
    if last_token in eos:
        return "stop"
    if count >= budget:
        return "context" if unlimited else "length"
    return "stop"


class ReasoningSplitter:
    """Preserve streamed text across delimiter boundaries without interpreting HTML."""
    markers = ("<think>","</think>")

    def __init__(self, enabled=False, prompt=""):
        self.enabled = enabled
        self.in_reasoning = enabled and prompt.rfind("<think>") > prompt.rfind("</think>")
        self.buffer = ""

    def feed(self, text):
        if not self.enabled:
            return [("answer",text)] if text else []
        self.buffer += text
        result = []
        while self.buffer:
            matches = [(self.buffer.find(marker),marker) for marker in self.markers if marker in self.buffer]
            if matches:
                index,marker = min(matches)
                if index:
                    result.append(("reasoning" if self.in_reasoning else "answer",self.buffer[:index]))
                self.in_reasoning = marker == "<think>"
                self.buffer = self.buffer[index+len(marker):]
                continue
            # Retain a suffix only if it could be the start of a split marker.
            held = max((length for marker in self.markers for length in range(1,len(marker))
                        if self.buffer.endswith(marker[:length])),default=0)
            safe = len(self.buffer)-held
            if safe:
                result.append(("reasoning" if self.in_reasoning else "answer",self.buffer[:safe]))
                self.buffer = self.buffer[safe:]
            break
        return result

    def finish(self):
        tail,self.buffer = self.buffer,""
        return [("reasoning" if self.in_reasoning else "answer",tail)] if tail else []
