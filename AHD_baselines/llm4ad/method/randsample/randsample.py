# Module Name: RandSample
# Last Revision: 2025/2/16
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Reference:
#   -  Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
#       Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
#       with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
# 
# Permission is granted to use the LLM4AD platform for research purposes. 
# All publications, software, or other works that utilize this platform 
# or any part of its codebase must acknowledge the use of "LLM4AD" and 
# cite the following reference:
# 
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
# 
# For inquiries regarding commercial use or licensing, please contact 
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

import concurrent.futures
import copy
import time
import traceback
from threading import Thread
from typing import Optional, Literal

from .profiler import RandSampleProfiler
from ...base import *


class RandSample:
    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: RandSampleProfiler = None,
                 num_samplers: int = 4,
                 num_evaluators: int = 4,
                 max_sample_nums: Optional[int] = 20,
                 *,
                 resume_mode: bool = False,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: Literal['thread', 'process'] = 'thread',
                 **kwargs):
        """Random Sampling
        Args:
            llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
            evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
            profiler        : an instance of 'llm4ad.method.randsample.RandSampleProfiler'. If you do not want to use it, you can pass a 'None'.
            max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not).
            num_samplers    : number of independent Samplers in the experiment.
            num_evaluators  : number of independent program Evaluators in the experiment.
            resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
            debug_mode      : if set to True, we will print detailed information.
            multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
                multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
                setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
                Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
                and you set this argument to 'thread'.
            **kwargs        : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.
        """
        # arguments and keywords
        self._template_program_str = evaluation.template_program
        self._max_sample_nums = max_sample_nums
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._debug_mode = debug_mode
        self._resume_mode = resume_mode

        # function to be evolved
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(self._template_program_str)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(self._template_program_str)

        # sampler, and evaluator
        self._sampler = SampleTrimmer(llm)
        llm.debug_mode = debug_mode
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler

        # statistics
        self._tot_sample_nums = 0

        # multi-thread executor for evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self._num_evaluators
            )

        # threads for sampling
        self._sampler_threads = [
            Thread(target=self._sample_evaluate_register) for _ in range(self._num_samplers)
        ]

        # self.prompt
        self._prompt_content = self._get_prompt()

        # pass parameters to profiler
        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)  # ZL: necessary

    def _get_prompt(self) -> str:
        template = copy.deepcopy(self._template_program)
        template.functions[0].name += '_v0'
        func_to_be_complete = copy.deepcopy(self._function_to_evolve)
        func_to_be_complete.name = self._function_to_evolve_name + '_v1'
        func_to_be_complete.docstring = f'  """Improved version of \'{self._function_to_evolve_name}_v0\'."""'
        func_to_be_complete.body = ''
        return '\n'.join([str(template), str(func_to_be_complete)])

    def _sample_evaluate_register(self):
        while (self._max_sample_nums is None) or (self._tot_sample_nums < self._max_sample_nums):
            try:
                # do sample
                draw_sample_start = time.time()
                sampled_funcs = self._sampler.draw_samples([self._prompt_content])
                draw_sample_times = time.time() - draw_sample_start
                avg_time_for_each_sample = draw_sample_times / len(sampled_funcs)
                
                # capture token counts
                prompt_tokens = getattr(self._sampler.llm, 'last_prompt_tokens', None)
                completion_tokens = getattr(self._sampler.llm, 'last_completion_tokens', None)
                thinking_tokens = getattr(self._sampler.llm, 'last_thinking_tokens', 0)

                # convert to program instance
                programs_to_be_eval = []
                for func in sampled_funcs:
                    program = SampleTrimmer.sample_to_program(func, self._template_program)
                    # if sample to program success
                    if program is not None:
                        programs_to_be_eval.append(program)

                # submit tasks to the thread pool and evaluate
                futures = []
                for program in programs_to_be_eval:
                    future = self._evaluation_executor.submit(self._evaluator.evaluate_program_record_time, program)
                    futures.append(future)
                # get evaluate scores, objectives, failure types, and evaluate times
                results = [f.result() for f in futures]
                scores = [i[0] for i in results]
                objectives_dicts = [i[1] for i in results]
                failure_types = [i[2] for i in results]
                times = [i[3] for i in results]

                # register to program database
                for program, score, objectives_dict, failure_type, eval_time in zip(programs_to_be_eval, scores, objectives_dicts, failure_types, times):
                    function = TextFunctionProgramConverter.program_to_function(program)
                    # check if the function has converted to Function instance successfully
                    if function is None:
                        continue
                    # register to profiler
                    if self._profiler:
                        function.score = score
                        function.objectives = objectives_dict
                        function.failure_type = failure_type
                        function.evaluate_time = eval_time
                        function.sample_time = avg_time_for_each_sample
                        function.prompt_time = avg_time_for_each_sample
                        function.prompt_tokens = prompt_tokens
                        function.completion_tokens = completion_tokens
                        function.thinking_tokens = thinking_tokens
                        self._profiler.register_function(function)
                    # update
                    self._tot_sample_nums += 1
            except KeyboardInterrupt:
                break
            except Exception as e:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue

        # shutdown evaluation_executor
        try:
            self._evaluation_executor.shutdown(cancel_futures=True)
        except:
            pass

    def run(self):
        if not self._resume_mode:
            # evaluate the template program, make sure the score of which is not 'None'
            score, objectives_dict, failure_type, eval_time = self._evaluator.evaluate_program_record_time(program=self._template_program)
            if score is None:
                raise RuntimeError('The score of the template function must not be "None".')

            # register the template program to the program database
            if self._profiler:
                self._function_to_evolve.score = score
                self._function_to_evolve.objectives = objectives_dict
                self._function_to_evolve.evaluate_time = eval_time
                self._profiler.register_function(self._function_to_evolve, program=str(self._template_program))

        # start sampling using multiple threads
        for t in self._sampler_threads:
            t.start()

        # join all threads to the main thread
        for t in self._sampler_threads:
            t.join()

        if self._profiler is not None:
            self._profiler.finish()

        self._sampler.llm.close()
