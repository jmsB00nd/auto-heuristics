import re
from typing import List, Dict, Optional, Tuple

class IdeaParser:
    """Extracts structured ideas and code blocks from raw LLM responses."""
    
    @staticmethod
    def extract_code(response_text: str, target_func: str = "qlosure_poly_heuristic") -> Optional[str]:
        patterns = [
            r"```python\s*\n(.*?)```",       
            r"```python(.*?)```",             
            r"```py\s*\n(.*?)```",            
            rf"```\s*\n(def {target_func}.*?)```",  
            rf"(def {target_func}\(self.*?\):.*?)(?:```|\Z)",  
        ]
        for pattern in patterns:
            code_match = re.search(pattern, response_text, re.DOTALL)
            if code_match:
                candidate = code_match.group(1).strip()
                if target_func in candidate or 'def ' in candidate:
                    return IdeaParser._sanitize_code(candidate, target_func)
        return None

    @staticmethod
    def parse_ideas(response_text: str) -> Tuple[List[Dict], List[Dict]]:
        """Applies multiple parsing strategies to extract ideas."""
        kept, eliminated = IdeaParser._strategy_idea_blocks(response_text)
        
        if not kept and not eliminated:
            kept, eliminated = IdeaParser._strategy_separated_blocks(response_text)
            
        if not kept and not eliminated:
            kept, eliminated = IdeaParser._strategy_numbered_lists(response_text)

        if not kept and not eliminated:
            kept = IdeaParser._create_fallback_idea(response_text)

        return kept, eliminated

    @staticmethod
    def _sanitize_code(code: str, target_func: str = "qlosure_poly_heuristic") -> str:
        """Strips out wrapper classes and returns the raw function."""
        lines = code.split('\n')
        func_start, func_indent = None, 0
        
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(f'def {target_func}'):
                func_start = i
                func_indent = len(line) - len(stripped)
                break

        if func_start is None:
            return code

        func_lines = [lines[func_start]]
        for j in range(func_start + 1, len(lines)):
            line = lines[j]
            if line.strip() == '':
                func_lines.append(line)
                continue
            if len(line) - len(line.lstrip()) <= func_indent:
                break
            func_lines.append(line)

        dedented = [l[func_indent:] if len(l) >= func_indent else l.lstrip() for l in func_lines]
        import_lines = [l for l in lines[:func_start] if l.strip().startswith(('import ', 'from '))]
        
        return '\n'.join(import_lines + [''] + dedented if import_lines else dedented)

    EXTRA_FIELDS = [
        'CORE_IDEA',
        'HOW_IT_WORKS',
        'WHY_IT_REDUCES_SWAPS',
        'TIME_COMPLEXITY',
        'STRENGTHS',
        'LIMITATIONS',
        'COLLISION_PREVENTION',
    ]

    @staticmethod
    def _extract_field(block: str, field: str) -> str:
        pattern = rf'{field}:\s*(.*?)(?=\n\s*(?:[A-Z_]{{3,}}\s*:|---)|\Z)'
        m = re.search(pattern, block, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _parse_single_block(block: str) -> Dict:
        """Parses a single textual block into an idea dictionary."""
        name_m = re.search(r'(?:IDEA_)?NAME:\s*(.+)', block, re.IGNORECASE) or re.search(r'^\s*\*\*(.+?)\*\*', block, re.MULTILINE)
        status_m = re.search(r'STATUS:\s*(\S+)', block, re.IGNORECASE)

        status = status_m.group(1).strip().upper() if status_m else "KEPT"
        if any(kw in status for kw in ['ELIM', 'REJECT']):
            status = 'ELIMINATED'

        idea = {
            'name': name_m.group(1).strip().strip('*') if name_m else "Unknown",
            'status': status,
            'description': IdeaParser._extract_field(block, 'DESCRIPTION'),
            'elimination_reason': "N/A",
        }
        for field in IdeaParser.EXTRA_FIELDS:
            idea[field.lower()] = IdeaParser._extract_field(block, field)
        return idea

    @staticmethod
    def _strategy_idea_blocks(text: str) -> Tuple[List[Dict], List[Dict]]:
        kept, eliminated = [], []
        for block in re.findall(r'IDEA:\s*(.*?)END_IDEA', text, re.DOTALL | re.IGNORECASE):
            idea = IdeaParser._parse_single_block(block)
            (kept if idea['status'] == 'KEPT' else eliminated).append(idea)
        return kept, eliminated

    @staticmethod
    def _strategy_separated_blocks(text: str) -> Tuple[List[Dict], List[Dict]]:
        kept, eliminated = [], []
        for block in re.split(r'\n---+\n', text):
            if re.search(r'(?:IDEA_NAME|NAME):\s*(.+)', block, re.IGNORECASE):
                idea = IdeaParser._parse_single_block(block)
                (kept if idea['status'] == 'KEPT' else eliminated).append(idea)
        return kept, eliminated
    
    @staticmethod
    def _strategy_numbered_lists(text: str) -> Tuple[List[Dict], List[Dict]]:
        kept, eliminated = [], []
        items = re.findall(r'(?:^|\n)\s*\d+\.\s*(?:\*\*)?([^*\n]+?)(?:\*\*)?\s*(?:\n|$)(.*?)(?=(?:\n\s*\d+\.|\Z))', text, re.DOTALL)
        for name, body in items:
            idea = IdeaParser._parse_single_block(f"NAME: {name}\n{body}")
            (kept if idea['status'] == 'KEPT' else eliminated).append(idea)
        return kept, eliminated

    @staticmethod
    def _create_fallback_idea(text: str) -> List[Dict]:
        return [{
            'name': 'PARSING_FAILED_manual_review_needed',
            'description': 'Failed to parse ideas. Raw response saved. Manual review required.',
            'status': 'KEPT',
            'raw_response_preview': text[:500],
        }]