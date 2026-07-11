# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2025/2/16
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

import os
import re
import sys
from typing import Literal, Optional, List, Tuple

import numpy as np
import pytz
import json
import logging
from threading import Lock
from datetime import datetime

from ...base import Function


class ProfilerBase:

    def __init__(self,
                 log_dir: Optional[str] = None,
                 *,
                 initial_num_samples=0,
                 log_style: Literal['simple', 'complex'] = 'complex',
                 create_random_path=True,
                 num_objs=1,
                 **kwargs):
        """Base profiler for recording experimental results.
        Args:
            log_dir            : the directory of current run
            initial_num_samples: the sample order start with `initial_num_samples`.
            create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        """
        assert log_style in ['simple', 'complex']

        from ... import TIMEZONE

        self._num_objs = num_objs
        self._num_samples = initial_num_samples
        self._process_start_time = datetime.now(TIMEZONE)
        self._result_folder = self._process_start_time.strftime('%Y%m%d_%H%M%S')

        self._log_dir = log_dir
        self._log_style = log_style
        self._cur_best_function = None if self._num_objs < 2 else [None for _ in range(self._num_objs)]
        self._cur_best_program_sample_order = None if self._num_objs < 2 else [None for _ in range(self._num_objs)]
        self._cur_best_program_score = float('-inf') if self._num_objs < 2 else [float('-inf') for _ in
                                                                                 range(self._num_objs)]
        self._cur_worst_function = None if self._num_objs < 2 else [None for _ in range(self._num_objs)]
        self._cur_worst_program_sample_order = None if self._num_objs < 2 else [None for _ in range(self._num_objs)]
        self._cur_worst_program_score = float('inf') if self._num_objs < 2 else [float('inf') for _ in
                                                                                 range(self._num_objs)]
        self._cur_best_program_data = None
        self._cur_worst_program_data = None
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._evaluate_timeout_program_num = 0  # Track timeouts separately
        self._evaluate_error_program_num = 0    # Track errors separately
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        self._tot_prompt_time = 0
        self._tot_token_count = 0
        self._tot_input_token_count = 0
        self._tot_output_token_count = 0
        self._tot_thinking_token_count = 0

        self._parameters = None
        self._logger_txt = logging.getLogger('root')

        if create_random_path:
            self._log_dir = os.path.join(
                log_dir,
                self._result_folder
            )
        else:
            self._log_dir = log_dir

        # lock for multi-thread invoking self.register_function(...)
        self._register_function_lock = Lock()

    def record_parameters(self, llm, prob, method):
        self._parameters = [llm, prob, method]
        self._create_log_path()

    def register_function(self, function: Function, program: str = '', *, resume_mode=False):
        """Record an obtained function and accumulate token counts."""
        try:
            self._register_function_lock.acquire()
            self._num_samples += 1

            # Accumulate token counts (input + output + thinking)
            inp = function.prompt_tokens or 0
            out = function.completion_tokens or 0
            think = function.thinking_tokens or 0
            self._tot_input_token_count += inp
            self._tot_output_token_count += out
            self._tot_thinking_token_count += think
            self._tot_token_count += inp + out

            self._record_and_print_verbose(function, program, resume_mode=resume_mode)
            if not resume_mode:
                self._write_json(function, program)
        finally:
            self._register_function_lock.release()

    def finish(self):
        """Generate a statistics summary file with execution metrics."""
        if not self._log_dir:
            return
        
        from ... import TIMEZONE
        
        # Calculate total execution time
        total_execution_time = (datetime.now(TIMEZONE) - self._process_start_time).total_seconds()
        
        # Calculate average score and objectives from successful evaluations
        average_score = None
        successful_scores = []
        objectives_by_key = {}  # {objective_1: [values], objective_2: [values], ...}
        
        # Try to read samples files and collect scores and objectives from successful evaluations
        samples_dir = os.path.join(self._log_dir, 'samples')
        if os.path.exists(samples_dir):
            try:
                # Read all samples files
                for filename in os.listdir(samples_dir):
                    if filename.endswith('.json'):
                        samples_file_path = os.path.join(samples_dir, filename)
                        with open(samples_file_path, 'r') as f:
                            samples = json.load(f)
                            if isinstance(samples, list):
                                for sample in samples:
                                    # Collect only non-null scores (successful evaluations)
                                    if 'score' in sample and sample['score'] is not None:
                                        successful_scores.append(sample['score'])
                                    
                                    # Collect objectives if present
                                    if 'objectives' in sample and sample['objectives'] is not None:
                                        for obj_key, obj_val in sample['objectives'].items():
                                            if obj_key not in objectives_by_key:
                                                objectives_by_key[obj_key] = []
                                            if obj_val is not None:
                                                objectives_by_key[obj_key].append(obj_val)
                
                # Calculate average score
                if successful_scores:
                    average_score = round(sum(successful_scores) / len(successful_scores), 2)
            except Exception as e:
                self._logger_txt.warning(f"Could not calculate average score/objectives: {e}")
        
        # Calculate average objectives (sorted by key for consistent output)
        average_objectives = {}
        for obj_key in sorted(objectives_by_key.keys()):
            values = objectives_by_key[obj_key]
            if values:
                average_objectives[f'average_{obj_key}'] = round(sum(values) / len(values), 2)
        
        # Create statistics dictionary
        statistics = {
            'total_execution_time_seconds': round(total_execution_time, 2),
            'total_evaluation_time_seconds': round(self._tot_evaluate_time, 2),
            'total_prompt_time_seconds': round(self._tot_prompt_time, 2),
            'total_sample_time_seconds': round(self._tot_sample_time, 2),
            'total_token_count': self._tot_token_count,
            'total_input_tokens': self._tot_input_token_count,
            'total_output_tokens': self._tot_output_token_count,
            'total_thinking_tokens': self._tot_thinking_token_count,
            'total_samples_evaluated': self._num_samples,
            'generated_heuristics': self._num_samples,
            'successful_evaluations': self._evaluate_success_program_num,
            'failed_evaluations': self._evaluate_failed_program_num,
            'timeout_evaluations': self._evaluate_timeout_program_num,
            'error_evaluations': self._evaluate_error_program_num,
            'best_score': float(self._cur_best_program_score) if self._num_objs < 2 else [float(s) for s in self._cur_best_program_score],
            'best_sample_order': self._cur_best_program_sample_order,
            'average_score': average_score,
        }
        
        # Add average objectives to statistics
        statistics.update(average_objectives)
        
        # Write statistics to JSON file
        stats_file_path = os.path.join(self._log_dir, 'run_statistics.json')
        with open(stats_file_path, 'w') as f:
            json.dump(statistics, f, indent=4)
        
        # Write best_heuristic.json
        if self._cur_best_program_data is not None:
            best_heuristic_path = os.path.join(self._log_dir, 'best_heuristic.json')
            with open(best_heuristic_path, 'w') as f:
                json.dump(self._cur_best_program_data, f, indent=4)
        
        # Write worst_heuristic.json
        if self._cur_worst_program_data is not None:
            worst_heuristic_path = os.path.join(self._log_dir, 'worst_heuristic.json')
            with open(worst_heuristic_path, 'w') as f:
                json.dump(self._cur_worst_program_data, f, indent=4)
        
        # Also log to run_log.txt
        if self._logger_txt:
            self._logger_txt.info('=====================================================================')
            self._logger_txt.info('EXECUTION STATISTICS')
            self._logger_txt.info('--------------------------------------------------------------------')
            self._logger_txt.info(f'Total Execution Time: {statistics["total_execution_time_seconds"]}s')
            self._logger_txt.info(f'Total Evaluation Time: {statistics["total_evaluation_time_seconds"]}s')
            self._logger_txt.info(f'Total Prompt Time: {statistics["total_prompt_time_seconds"]}s')
            self._logger_txt.info(f'Total Sample Time: {statistics["total_sample_time_seconds"]}s')
            self._logger_txt.info(f'Total Token Count: {statistics["total_token_count"]}')
            self._logger_txt.info(f'  - Input Tokens   : {statistics["total_input_tokens"]}')
            self._logger_txt.info(f'  - Output Tokens  : {statistics["total_output_tokens"]}')
            if statistics["total_thinking_tokens"]:
                self._logger_txt.info(f'  - Thinking Tokens: {statistics["total_thinking_tokens"]}')
            self._logger_txt.info(f'Total Samples Evaluated: {statistics["total_samples_evaluated"]}')
            self._logger_txt.info(f'Generated Heuristics: {statistics["generated_heuristics"]}')
            self._logger_txt.info(f'Successful Evaluations: {statistics["successful_evaluations"]}')
            self._logger_txt.info(f'Failed Evaluations: {statistics["failed_evaluations"]}')
            self._logger_txt.info(f'  - Timeout Evaluations: {statistics["timeout_evaluations"]}')
            self._logger_txt.info(f'  - Error Evaluations: {statistics["error_evaluations"]}')
            self._logger_txt.info(f'Best Score: {statistics["best_score"]}')
            self._logger_txt.info(f'Average Score (successful only): {statistics["average_score"]}')
            # Log average objectives
            for obj_key in sorted(objectives_by_key.keys()):
                avg_key = f'average_{obj_key}'
                if avg_key in statistics:
                    self._logger_txt.info(f'Average {obj_key}: {statistics[avg_key]}')
            self._logger_txt.info('=====================================================================')


    def get_logger(self):
        pass

    def resume(self, *args, **kwargs):
        pass

    def _write_json(self, function: Function, program: str, *, record_type: Literal['history', 'best'] = 'history',
                    record_sep=200):
        """Write function data to a JSON file.
        Args:
            function   : The function object containing score and string representation.
            record_type: Type of record, 'history' or 'best'. Defaults to 'history'.
            record_sep : Separator for history records. Defaults to 200.
        """
        if not self._log_dir:
            return

        sample_order = self._num_samples
        content = {
            'sample_order': sample_order,
            'function': str(function),
            'score': function.score,
            'operator': function.operator,
            'sample_time': function.sample_time,
            'prompt_time': function.prompt_time,
            'evaluate_time': function.evaluate_time,
            'prompt_tokens': function.prompt_tokens,
            'completion_tokens': function.completion_tokens,
            'thinking_tokens': function.thinking_tokens,
            'program': program,
        }
        
        # Add objectives if present
        if function.objectives is not None:
            content['objectives'] = function.objectives

        if record_type == 'history':
            lower_bound = ((sample_order - 1) // record_sep) * record_sep
            upper_bound = lower_bound + record_sep
            filename = f'samples_{lower_bound + 1}~{upper_bound}.json'
        else:
            filename = 'samples_best.json'

        path = os.path.join(self._samples_json_dir, filename)

        try:
            with open(path, 'r') as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(content)

        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def _record_and_print_verbose(self, function, program='', *, resume_mode=False):
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        prompt_time = function.prompt_time
        score = function.score
        operator = function.operator
        prompt_tokens = function.prompt_tokens
        completion_tokens = function.completion_tokens
        thinking_tokens = function.thinking_tokens

        # Prepare sample data for best/worst tracking
        sample_data = {
            'sample_order': self._num_samples,
            'function': str(function),
            'score': score,
            'operator': operator,
            'sample_time': sample_time,
            'prompt_time': prompt_time,
            'evaluate_time': evaluate_time,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'thinking_tokens': thinking_tokens,
            'program': program,
        }

        # update best function
        if self._num_objs < 2:
            if score is not None and score > self._cur_best_program_score:
                self._cur_best_function = function
                self._cur_best_program_score = score
                self._cur_best_program_sample_order = self._num_samples
                self._cur_best_program_data = sample_data
                if not resume_mode:
                        self._write_json(function, record_type='best', program=program)
            # update worst function
            if score is not None and score < self._cur_worst_program_score:
                self._cur_worst_function = function
                self._cur_worst_program_score = score
                self._cur_worst_program_sample_order = self._num_samples
                self._cur_worst_program_data = sample_data
        else:
            if score is not None:
                for i in range(self._num_objs):
                    if score[i] > self._cur_best_program_score[i]:
                        self._cur_best_function[i] = function
                        self._cur_best_program_score[i] = score[i]
                        self._cur_best_program_sample_order[i] = self._num_samples
                        if not resume_mode:
                                self._write_json(function, record_type='best', program=program)
                    # update worst function
                    if score[i] < self._cur_worst_program_score[i]:
                        self._cur_worst_function[i] = function
                        self._cur_worst_program_score[i] = score[i]
                        self._cur_worst_program_sample_order[i] = self._num_samples

        if not resume_mode:
            # log attributes of the function
            if self._log_style == 'complex':
                print(f'================= Evaluated Function =================')
                print(f'{function_str}')
                print(f'------------------------------------------------------')
                print(f'Operator     : {operator}')
                print(f'Score        : {str(score)}')
                print(f'Sample time  : {str(sample_time)}')
                if prompt_time is not None:
                    print(f'Prompt time  : {str(prompt_time)}')
                print(f'Evaluate time: {str(evaluate_time)}')
                if prompt_tokens is not None:
                    print(f'Input tokens  : {prompt_tokens}')
                if completion_tokens is not None:
                    print(f'Output tokens : {completion_tokens}')
                if thinking_tokens is not None and thinking_tokens:
                    print(f'Think tokens  : {thinking_tokens}')
                print(f'Sample orders: {str(self._num_samples)}')
                print(f'------------------------------------------------------')
                print(f'Current best score: {self._cur_best_program_score}')
                print(f'======================================================\n')
            else:
                if score is None:
                    if self._num_objs < 2:
                        print(
                            f'Sample{self._num_samples}: Score=None    Cur_Best_Score={self._cur_best_program_score: .3f}')
                    else:
                        # Format the list of best scores dynamically
                        best_scores_str = ", ".join([f"{s: .3f}" for s in self._cur_best_program_score])
                        print(
                            f'Sample{self._num_samples}: Score=None    Cur_Best_Score=[{best_scores_str}]')
                else:
                    if self._num_objs < 2:
                        print(
                            f'Sample{self._num_samples}: Score={score: .3f}     Cur_Best_Score={self._cur_best_program_score: .3f}')
                    else:
                        # Format both current scores and best scores dynamically
                        scores_str = ", ".join([f"{s: .3f}" for s in score])
                        best_scores_str = ", ".join([f"{s: .3f}" for s in self._cur_best_program_score])
                        print(
                            f'Sample{self._num_samples}: Score=[{scores_str}]     Cur_Best_Score=[{best_scores_str}]')

        # update statistics about function
        if score is not None:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1
            # Track specific failure type
            failure_type = getattr(function, 'failure_type', None)
            if failure_type == 'timeout':
                self._evaluate_timeout_program_num += 1
            elif failure_type == 'error':
                self._evaluate_error_program_num += 1

        if sample_time is not None:
            self._tot_sample_time += sample_time

        if prompt_time is not None:
            self._tot_prompt_time += prompt_time

        if evaluate_time:
            self._tot_evaluate_time += evaluate_time

    def _create_log_path(self):
        self._samples_json_dir = os.path.join(self._log_dir, 'samples')
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(self._samples_json_dir, exist_ok=True)

        file_name = self._log_dir + '/run_log.txt'
        file_mode = 'a' if os.path.isfile(file_name) else 'w'

        self._logger_txt.setLevel(level=logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s', '%Y-%m-%d %H:%M:%S')

        for hdlr in self._logger_txt.handlers[:]:
            self._logger_txt.removeHandler(hdlr)

        # add handler
        fileout = logging.FileHandler(file_name, mode=file_mode)
        fileout.setLevel(logging.INFO)
        fileout.setFormatter(formatter)
        self._logger_txt.addHandler(fileout)
        self._logger_txt.addHandler(logging.StreamHandler(sys.stdout))

        # write initial parameters
        llm = self._parameters[0]
        prob = self._parameters[1]
        method = self._parameters[2]

        self._logger_txt.info('====================================================================')
        self._logger_txt.info('LLM Parameters')
        self._logger_txt.info('--------------------------------------------------------------------')
        self._logger_txt.info(f'  - LLM: {llm.__class__.__name__}')
        for attr, value in llm.__dict__.items():
            if attr not in ['_functions']:
                self._logger_txt.info(f'  - {attr}: {value}')
        self._logger_txt.info('====================================================================')
        self._logger_txt.info('Problem Parameters')
        self._logger_txt.info('--------------------------------------------------------------------')
        self._logger_txt.info(f'  - Problem: {prob.__class__.__name__}')
        for attr, value in prob.__dict__.items():
            if attr not in ['template_program', '_datasets']:
                self._logger_txt.info(f'  - {attr}: {value}')

        self._logger_txt.info('====================================================================')
        self._logger_txt.info('Method Parameters')
        self._logger_txt.info('--------------------------------------------------------------------')
        self._logger_txt.info(f'  - Method: {method.__class__.__name__}')
        for attr, value in method.__dict__.items():
            if attr not in ['llm', '_evaluator', '_profiler', '_template_program_str', '_template_program',
                            '_function_to_evolve', '_population', '_sampler', '_task_description_str']:
                self._logger_txt.info(f'  - {attr}: {value}')

        self._logger_txt.info('=====================================================================')

    @classmethod
    def load_logfile(cls, logdir, valid_only=False) -> Tuple[List[str], List[float]]:
        file_dir = os.path.join(logdir, 'samples')
        # get all file directories
        all_files = os.listdir(file_dir)
        # filer `samples_*.json` files and ignore `samples_best.json`
        sample_files = [f for f in all_files if f.startswith('samples_') and f != 'samples_best.json']

        def extract_number(filename):
            # match the first number of the filename
            match = re.search(r'samples_(\d+)~', filename)
            if match:
                return int(match.group(1))
            return 0

        sorted_files = sorted(sample_files, key=extract_number)

        all_func = []
        all_score = []
        all_algorithm = []

        for file in sorted_files:
            file_path = os.path.join(file_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    samples = json.load(f)
                except Exception as e:
                    print(e)
                    print(file_path)
            for sample in samples:
                func = sample['function']
                acc = sample['score'] if sample['score'] else float('-inf')
                if valid_only:
                    if acc is None or np.isinf(acc):
                        continue
                    all_func.append(func)
                    all_score.append(acc)
                else:
                    all_func.append(func)
                    all_score.append(acc)
                if 'algorithm' in sample:
                    all_algorithm.append(sample['algorithm'])

        return all_func, all_score
