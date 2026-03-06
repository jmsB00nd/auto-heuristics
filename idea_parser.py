import re
from typing import List, Dict, Optional, Tuple

class IdeaParser:
    """Extracts structured ideas and code blocks from raw LLM responses."""
    
    @staticmethod
    def extract_code(response_text: str) -> Optional[str]:
        patterns = [
            r"```python\s*\n(.*?)```",       
            r"```python(.*?)```",             
            r"```py\s*\n(.*?)```",            
            r"```\s*\n(def qlosure_poly_heuristic.*?)```",  
            r"(def qlosure_poly_heuristic\(self,\s*swap_gate\):.*?)(?:```|\Z)",  
        ]
        for pattern in patterns:
            code_match = re.search(pattern, response_text, re.DOTALL)
            if code_match:
                candidate = code_match.group(1).strip()
                if 'qlosure_poly_heuristic' in candidate or 'def ' in candidate:
                    return IdeaParser._sanitize_code(candidate)
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

        kept.sort(key=lambda x: x.get('average_score', 0), reverse=True)
        return kept, eliminated

    @staticmethod
    def _sanitize_code(code: str) -> str:
        """Strips out wrapper classes and returns the raw function."""
        lines = code.split('\n')
        func_start, func_indent = None, 0
        
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith('def qlosure_poly_heuristic'):
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

    @staticmethod
    def _parse_single_block(block: str) -> Dict:
        """Parses a single textual block into an idea dictionary."""
        name_m = re.search(r'(?:IDEA_)?NAME:\s*(.+)', block, re.IGNORECASE) or re.search(r'^\s*\*\*(.+?)\*\*', block, re.MULTILINE)
        status_m = re.search(r'STATUS:\s*(\S+)', block, re.IGNORECASE)
        desc_m = re.search(r'DESCRIPTION:\s*(.*?)(?=\n\s*(?:[A-Z_]{3,}\s*:|---)|\Z)', block, re.IGNORECASE | re.DOTALL)
        avg_m = re.search(r'AVERAGE[_\s]*SCORE:\s*(\d+(?:\.\d+)?)', block, re.IGNORECASE)
        
        status = status_m.group(1).strip().upper() if status_m else "KEPT"
        if any(kw in status for kw in ['ELIM', 'REJECT']):
            status = 'ELIMINATED'

        return {
            'name': name_m.group(1).strip().strip('*') if name_m else "Unknown",
            'status': status,
            'description': desc_m.group(1).strip() if desc_m else "",
            'average_score': float(avg_m.group(1)) if avg_m else 5.0,
            'elimination_reason': "N/A"
        }

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
            'average_score': 1.0,
            'raw_response_preview': text[:500],
        }]