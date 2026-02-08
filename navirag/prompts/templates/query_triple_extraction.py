from ...utils.llm_utils import convert_format_to_template

query_re_system = """Your task is to translate a natural language question into a list of structured knowledge graph queries (triples).
These triples will be used to retrieve information to answer the question.

Guidelines:
1. Use the provided Named Entities as the Subject or Object of the triples.
2. For the missing information being asked (e.g. "Who", "Where", "When"), use the symbol "?".
3. Break down multi-hop questions into a chain of triples.
4. Extract relationships like "director", "born in", "spouse", "author", etc.
"""

query_re_frame = """Convert the question into a JSON dict containing a 'triples' list.
Question:
```
{passage}
```
{named_entity_json}
"""

query_re_input_1 = query_re_frame.format(
    passage="Which film has the director died earlier, Swat The Spy or The Crimson City?",
    named_entity_json='{"named_entities": ["Swat The Spy", "The Crimson City"]}'
)

query_re_output_1 = """{"triples": [
    ["Swat The Spy", "director", "?"],
    ["The Crimson City", "director", "?"],
    ["?", "death date", "?"]
]}
"""

query_re_input_2 = query_re_frame.format(
    passage="Who is the director of the movie starred by Tom Hanks?",
    named_entity_json='{"named_entities": ["Tom Hanks"]}'
)

query_re_output_2 = """{"triples": [
    ["Tom Hanks", "starred in", "?"],
    ["?", "director", "?"]
]}
"""

prompt_template = [
    {"role": "system", "content": query_re_system},
    {"role": "user", "content": query_re_input_1},
    {"role": "assistant", "content": query_re_output_1},
    {"role": "user", "content": query_re_input_2},
    {"role": "assistant", "content": query_re_output_2},
    {"role": "user", "content": convert_format_to_template(original_string=query_re_frame, placeholder_mapping=None, static_values=None)}
]