from __future__ import annotations

import re
import time
from typing import Tuple, List, Dict

from .prompt import EoHPrompt
from ...base import LLM, SampleTrimmer, Function, Program
from ...base.modify_code import ModifyCode


class EoHSampler:
    def __init__(self, llm: LLM, template_program: str | Program):
        self.llm = llm
        self._template_program = template_program

    def get_thought_and_function(self, prompt: str) -> Tuple[str, Function, float]:
        """Get thought and function from prompt, returning (thought, function, prompt_time)."""
        prompt_start = time.time()
        response = self.llm.draw_sample(prompt)
        prompt_time = time.time() - prompt_start
        
        # Capture token counts from LLM if available
        prompt_tokens = getattr(self.llm, 'last_prompt_tokens', None)
        completion_tokens = getattr(self.llm, 'last_completion_tokens', None)
        thinking_tokens = getattr(self.llm, 'last_thinking_tokens', 0)
        
        thought = self.__class__.trim_thought_from_response(response)
        code = SampleTrimmer.trim_preface_of_function(response)

        function = SampleTrimmer.sample_to_function(code, self._template_program)
        
        # Attach token counts to the function object
        if function is not None:
            function.prompt_tokens = prompt_tokens
            function.completion_tokens = completion_tokens
            function.thinking_tokens = thinking_tokens

        return thought, function, prompt_time

    @classmethod
    def trim_thought_from_response(cls, response: str) -> str | None:
        try:
            pattern = r'\{.*?\}'  # Compared with r'\{(.*)\}'
            bracketed_texts = re.findall(pattern, response)
            return bracketed_texts[0]
        except:
            return None
