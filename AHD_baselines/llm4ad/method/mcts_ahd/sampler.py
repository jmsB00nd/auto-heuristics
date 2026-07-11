from __future__ import annotations

import re
import time
from typing import Tuple, List, Dict

from .prompt import MAPrompt
from ...base import LLM, SampleTrimmer, Function, Program


class MASampler:
    def __init__(self, llm: LLM, template_program: str | Program):
        self.llm = llm
        self._template_program = template_program

    def get_thought_and_function(self, task_description: str, prompt: str) -> Tuple[str, Function, float]:
        """Get thought and function from prompt, returning (thought, function, prompt_time)."""
        prompt_start = time.time()
        response = self.llm.draw_sample(prompt)
        
        # Capture token counts from first LLM call
        prompt_tokens = getattr(self.llm, 'last_prompt_tokens', None)
        completion_tokens = getattr(self.llm, 'last_completion_tokens', None)
        thinking_tokens = getattr(self.llm, 'last_thinking_tokens', 0)
        
        thought = self.__class__.trim_thought_from_response(response)
        code = SampleTrimmer.trim_preface_of_function(response)
        function = SampleTrimmer.sample_to_function(code, self._template_program)
        if thought is None or function is None:
            prompt_time = time.time() - prompt_start
            return thought, function, prompt_time
        prompt2 = self.get_prompt_refine(task_description, thought, str(function))
        prompt_start2 = time.time()
        describe = self.llm.draw_sample(prompt2)
        prompt_time = (time.time() - prompt_start2) + (time.time() - prompt_start)
        
        # Accumulate token counts from second LLM call
        if prompt_tokens is not None and getattr(self.llm, 'last_prompt_tokens', None) is not None:
            prompt_tokens += getattr(self.llm, 'last_prompt_tokens', 0)
        if completion_tokens is not None and getattr(self.llm, 'last_completion_tokens', None) is not None:
            completion_tokens += getattr(self.llm, 'last_completion_tokens', 0)
        thinking_tokens += getattr(self.llm, 'last_thinking_tokens', 0)
        
        # Attach token counts to the function object
        if function is not None:
            function.prompt_tokens = prompt_tokens
            function.completion_tokens = completion_tokens
            function.thinking_tokens = thinking_tokens
        
        return describe, function, prompt_time

    def get_prompt_refine(self, task_prompt: str, idea: str, code: str):
        prompt_content = task_prompt + "\n" + "Following is the Design Idea of a heuristic algorithm for the problem and the code for implementing the heuristic algorithm.\n"
        prompt_content += "\nDesign Idea:\n" + idea
        prompt_content += "\n\nCode:\n" + code
        prompt_content += "\n\nThe content of the Design Idea idea cannot fully represent what the algorithm has done informative. So, now you should re-describe the algorithm using less than 3 sentences.\n"
        prompt_content += "Hint: You should reference the given Design Idea and highlight the most critical design ideas of the code. You can analyse the code to describe which variables are given higher priorities and which variables are given lower priorities, the parameters and the structure of the code."
        return prompt_content

    @classmethod
    def trim_thought_from_response(cls, response: str) -> str | None:
        try:
            pattern = r'\{(.*?)\}'  # Compared with r'\{(.*)\}'
            bracketed_texts = re.findall(pattern, response)
            return bracketed_texts[0]
        except:
            return None
