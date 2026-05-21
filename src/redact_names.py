from pathlib import Path
import pandas as pd
import re
from rapidfuzz import process, fuzz
from sharks_aliases import sharks_aliases

SHORT_ALIAS_THRESHOLD = 4
FUZZY_THRESHOLD = 85

WORD_PATTERN = re.compile(r"[a-zA-Z0-9']+")

def build_alias_lookup(player_aliases):
    alias_to_player = {}
    long_aliases = set()
    for player, aliases in player_aliases.items():
        for alias in aliases:
            alias = alias.lower().strip()
            alias_to_player[alias] = player
            if len(alias) > SHORT_ALIAS_THRESHOLD:
                long_aliases.add(alias)
    return alias_to_player, long_aliases


def redact_comment(comment, alias_to_player, long_aliases):
    if not isinstance(comment, str):
        return comment

    spans_to_redact = []

    for match in WORD_PATTERN.finditer(comment):
        word = match.group()
        word_lower = word.lower()

        lookup = word_lower[:-2] if word_lower.endswith("'s") else word_lower

        matched = False

        if lookup in alias_to_player:
            matched = True
        elif len(lookup) > SHORT_ALIAS_THRESHOLD:
            result = process.extractOne(
                lookup,
                long_aliases,
                scorer=fuzz.ratio,
                score_cutoff=FUZZY_THRESHOLD
            )
            if result:
                matched = True

        if matched:
            spans_to_redact.append((match.start(), match.end()))

    result = comment
    for start, end in reversed(spans_to_redact):
        result = result[:start] + "[PLAYER]" + result[end:]

    return result



if __name__ == "__main__":
    TRAINING_DATA_DIR = Path("data/training")

    alias_to_player, long_aliases = build_alias_lookup(sharks_aliases)

    # Load data
    input_file = TRAINING_DATA_DIR / "labeled_player_comments.csv"
    df = pd.read_csv(input_file, encoding="latin-1")
    df.columns = df.columns.str.strip()
    comment_col = "Comment"
    
    df["Redacted_Comment"] = df[comment_col].apply(
        lambda x: redact_comment(x, alias_to_player, long_aliases)
    )

    # Save to CSV
    output = df[["Redacted_Comment", "Sentiment"]].rename(columns={"Redacted_Comment": "Comment"})
    output_file = TRAINING_DATA_DIR / "redacted_training_comments.csv"
    output.to_csv(output_file, index=False)

    print(f"Saved {len(output)} rows to {output_file.name}")
