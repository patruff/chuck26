"""
Centralized system prompts for parody generation.

This module contains all prompts used by the parody generation system.
Edit these prompts to customize the model's behavior and output style.
"""

# =============================================================================
# AGENT SYSTEM PROMPT
# =============================================================================
# This is the base system prompt for the CodeAgent that generates parodies.
# It includes template placeholders that smolagents requires.

AGENT_SYSTEM_PROMPT = """Your role is to generate creative parodies. Your primary goal is HUMOR - phonetic similarity helps but should never come at the cost of being funny.
{{authorized_imports}}
import json
from smolagents import load_tool
{{managed_agents_descriptions}}
You have access to word_phone_tool to check phonetic similarity between words."""


# =============================================================================
# PARODY STYLE GUIDE
# =============================================================================
# Guidelines for creating effective parodies, derived from successful examples.

PARODY_STYLE_GUIDE = """
PARODY STYLE GUIDE (based on successful examples):

1. PHONETIC PRIORITY: The parody must sound like the original when spoken
   - "You Got Served" → "You Got Serbed" (nearly identical pronunciation)
   - Single consonant/vowel swaps work best

2. HUMOR THROUGH CONTRAST: Subvert expectations
   - Epic → Mundane: "The Matrix" → "The Mattress"
   - Serious → Silly: "Die Hard" → "Dye Hard"
   - Action → Domestic: "Kill Bill" → "Pill Bill"

3. EDGY IS GOOD: Don't shy away from adult themes
   - "Star Wars" → "Scar Whores" (pushes boundaries)
   - "The Office" → "The Orifice" (crude but funny)

4. COMPOUND WORDS: Replace parts while keeping structure
   - "The Godfather" → "The Codfather" (god→cod)
   - "Goodfellas" → "Foodfellas" (good→food)

5. DOUBLE CHANGES: Sometimes changing 2 words is funnier
   - "Pulp Fiction" → "Gulp Friction" (both words changed)
   - Creates completely new absurd meaning
"""


# =============================================================================
# GENERATION PROMPT TEMPLATE
# =============================================================================
# The main prompt template for generating parodies.
# Variables: {title}, {examples_text}, {style_guide}, {suggestions}

GENERATION_PROMPT_TEMPLATE = """Create a funny parody of the title: "{title}"

=== KNOWN FUNNY EXAMPLES ===
These parodies work well - learn from their style:
{examples_text}

{style_guide}

=== YOUR TASK ===
Create a parody for: "{title}"

=== STEP 1: WORD SUGGESTIONS ===
I've already gathered phonetically similar funny words for each word in the title.
Here are the suggestions (word → candidates with phonetic similarity scores):

{suggestions}

=== STEP 2: PHONETIC VERIFICATION ===
For each word you want to replace, you MUST verify it with the word_phone_tool.
The tool returns a similarity score from 0.0 to 1.0.
- Score > 0.6: ACCEPTABLE (sounds similar enough)
- Score < 0.6: REJECT (doesn't sound similar, try another word)

How to use the tool:
```
word_phone_tool("original_word", "replacement_word")
```

Example:
```
word_phone_tool("Served", "Serbed") → 0.92  ✓ Great match!
word_phone_tool("Matrix", "Mattress") → 0.82  ✓ Good match!
```

=== STEP 3: SYNTHESIZE THE BEST PARODY ===
Now combine the best replacements to create a funny parody.

<think>
BRAINSTORMING:
Look at the suggestions above. Which combinations would be:
1. Phonetically similar (score > 0.6)?
2. Actually funny to the target audience (edgy, adult humor)?
3. Create an absurd/unexpected meaning?

Let me try several combinations and verify each with the tool...

### Attempt 1:
**"[parody attempt 1]"**
Tool checks:
- word_phone_tool("[word1]", "[replacement1]") → [score]
- word_phone_tool("[word2]", "[replacement2]") → [score]
All scores > 0.6? [Yes/No]
Humor rating: X/10
Why it's funny: [explanation]

### Attempt 2:
**"[parody attempt 2]"**
Tool checks:
- word_phone_tool("[word1]", "[replacement1]") → [score]
- word_phone_tool("[word2]", "[replacement2]") → [score]
All scores > 0.6? [Yes/No]
Humor rating: X/10
Why it's funny: [explanation]

### Attempt 3:
**"[parody attempt 3]"**
Tool checks:
- word_phone_tool("[word1]", "[replacement1]") → [score]
All scores > 0.6? [Yes/No]
Humor rating: X/10
Why it's funny: [explanation]

SELECTION:
Compare the valid attempts (all scores > 0.6) and pick the FUNNIEST one.
</think>

=== FINAL OUTPUT ===

### Final Chosen Parody:
**"[Your chosen parody]"**

### Phonetic Verification Summary:
[List each word replacement with its score]
- "[original1]" → "[replacement1]": [score] ✓
- "[original2]" → "[replacement2]": [score] ✓

### Why This Works:
1. Phonetic similarity: [explain how it sounds like the original]
2. Humor factor: [explain why it's funny]
3. Style match: [compare to successful examples above]"""


# =============================================================================
# RLVR TRAINING PROMPTS
# =============================================================================
# Prompts for RLVR training data generation.

def get_rlvr_system_prompt(reasoning_start: str, reasoning_end: str,
                           solution_start: str, solution_end: str) -> str:
    """Generate RLVR system prompt with custom tags."""
    return f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start} and {solution_end}."""


RLVR_PARODY_INSTRUCTION = """Create a funny parody of the movie title '{title}'.
Use phonetic similarity checking to verify your word choices sound similar.
Show your reasoning process."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_generation_prompt(title: str, examples_text: str, suggestions_json: str) -> str:
    """
    Build the complete generation prompt for a given title.

    Args:
        title: The movie/show title to parody
        examples_text: Formatted examples of known funny parodies
        suggestions_json: JSON string of word suggestions

    Returns:
        Complete prompt string ready for the agent
    """
    return GENERATION_PROMPT_TEMPLATE.format(
        title=title,
        examples_text=examples_text,
        style_guide=PARODY_STYLE_GUIDE,
        suggestions=suggestions_json
    )
