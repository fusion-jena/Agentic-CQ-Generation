import json
from utils.llm_client import CustomOllamaClient
from utils.context_manager import ContextManager

# Metadata is always near the start of a paper (title, authors, abstract, keywords).
# We cap the excerpt to this limit regardless of available context.
_METADATA_CHAR_LIMIT = 5_000


class PaperMetadataExtractor:
    def __init__(self, model_name="deepseek-r1:32b"):
        self.model_name = model_name
        self.llm = CustomOllamaClient(model=model_name)

    def extract(self, paper_text: str, source_filename: str = "") -> dict:
        ctx     = ContextManager(self.model_name, prompt_overhead_tokens=400, response_budget_tokens=300, llm_client=self.llm)
        # Take only the beginning of the paper — all bibliographic info lives there.
        excerpt = paper_text[:min(_METADATA_CHAR_LIMIT, ctx.available_chars)]

        prompt = f"""
You are a scientific literature expert. Extract bibliographic metadata from the research paper below.

PAPER TEXT (opening section):
{excerpt}

Extract each field. If a field cannot be found, use null.

OUTPUT FORMAT (JSON only, no explanation, no markdown fences):
{{
  "title": "Full paper title",
  "first_author_lastname": "Last name of the first author only",
  "year": 2023,
  "journal": "Journal or conference name",
  "doi": "10.xxxx/xxxxx or null",
  "keywords": ["keyword1", "keyword2"],
  "polymer_systems": ["PMMA", "PS-b-PMMA"],
  "polymerization_methods": ["RAFT", "ATRP"],
  "paper_type": "one of: research article | review article | methods paper | perspective | letter | communication | book chapter | other"
}}
"""
        response = self.llm.invoke(prompt)
        metadata = self._parse(response)

        author = (metadata.get("first_author_lastname") or "Unknown").strip()
        year   = str(metadata.get("year") or "0000").strip()
        metadata["paper_id"]        = f"{author}{year}"
        metadata["source_filename"] = source_filename
        return metadata

    def drain_stats(self) -> list[dict]:
        return self.llm.drain_stats()

    def _parse(self, response: str) -> dict:
        try:
            start = response.find('{')
            end   = response.rfind('}') + 1
            if start != -1 and end != -1:
                return json.loads(response[start:end])
        except Exception:
            pass
        return {
            "title": None,
            "first_author_lastname": "Unknown",
            "year": "0000",
            "journal": None,
            "doi": None,
            "keywords": [],
            "polymer_systems": [],
            "polymerization_methods": [],
        }
