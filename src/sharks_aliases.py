import pandas as pd
import re
from rapidfuzz import process, fuzz

sharks_aliases = {
    "A. Gaudette": ["adam", "gaudette", "gaud", "gauds"],
    "A. Nedeljkovic": ["nedeljkovic", "ned"],
    "A. Wennberg": ["alex", "wennberg", "wenny", "wennie"],
    "B. Goodrow": ["barclay", "goodrow", "goody", "goodie"],
    "C. Graf": ["collin", "graf", "graffer"],
    "D. Orlov": ["dmitry", "orlov", "orly", "orlie", "scorlov"],
    "I. Chernyshov": ["igor", "chernyshov", "cherny", "chernie"],
    "J. Klingberg": ["klingberg", "klinger", "johnberg", "kling", "klankberg", "dingleberg", "klongborg", "klingfuck"],
    "K. Sherwood": ["kiefer", "sherwood", "kief"],
    "L. Brossoit": ["laurent", "brossoit"],
    "L. Cagnoni": ["luca", "cagnoni", "cags"],
    "M. Celebrini": ["macklin", "celebrini", "mack", "celly", "cellie", "macky", "mackie"],
    "M. Ferraro": ["mario", "ferraro", "mar"],
    "M. Misa": ["mike", "michael", "misa", "mise", "mis"],
    "N. Leddy": ["nick", "leddy", "leddie", "leds"],
    "P. Regenda": ["pavol", "regenda", "reggy", "reggie"],
    "P. Kurashev": ["philipp", "kurashev", "phil", "chevy", "chevie"],
    "R. Reaves": ["ryan", "reaves", "reavo"],
    "S. Dickinson": ["sam", "dickinson", "dicky", "dickie", "dick6"],
    "S. Mukhamadullin": ["shakir", "mukhamadullin", "muk", "shak"],
    "T. Dellandrea": ["ty", "dellandrea", "drea", "delly", "dellie"],
    "T. Toffoli": ["tyler", "toffoli", "toff"],
    "V. Desharnais": ["vincent", "desharnais", "vinny", "vinnie"],
    "W. Smith": ["will", "smith", "smitty", "smittie"],
    "W. Eklund": ["william", "eklund", "eky", "ekie", "ekky", "gecko"],
    "Y. Askarov": ["yaroslav", "askarov", "asky", "askie"],
    "Z. Ostapchuk": ["zack", "ostapchuk", "ostap", "chucky"],
}

# Set short alias threshold for exact matching
SHORT_ALIAS_THRESHOLD = 4
FUZZY_THRESHOLD = 85  # higher = stricter

WORD_PATTERN = re.compile(r"[a-zA-Z0-9']+")
REPEATED_CHAR_PATTERN = re.compile(r'(.)\1{2,}')


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


def find_players_in_comment(comment, alias_to_player, long_aliases):
    words = WORD_PATTERN.findall(str(comment).lower())
    found = {}

    for word in words:
        if word.endswith("'s"):
            word = word[:-2]
        
        # For words that have repeated chars reduce to 2 occurrences ("mackkkkkk" -> "mackk")
        word = REPEATED_CHAR_PATTERN.sub(r'\1\1', word) 

        matched_alias = None
        player = None

        # Exact match first
        if word in alias_to_player:
            matched_alias = word
            player = alias_to_player[word]
        elif word[:-1] in alias_to_player:
            matched_alias = word[:-1]
            player = alias_to_player[matched_alias]

        # Fuzzy match only for longer words
        elif len(word) > SHORT_ALIAS_THRESHOLD:
            result = process.extractOne(
                word,
                long_aliases,
                scorer=fuzz.ratio,
                score_cutoff=FUZZY_THRESHOLD
            )
            if result:
                matched_alias = result[0]
                player = alias_to_player[matched_alias]

        if player:
            found.setdefault(player, set()).add(matched_alias)

    return found



if __name__ == "__main__":
    alias_to_player, long_aliases = build_alias_lookup(sharks_aliases)

    # Load data
    df = pd.read_csv("reddit_top_comments.csv")
    df.columns = df.columns.str.strip()
    comment_col = "Top Level Comment"
    
    df["players_mentioned"] = df[comment_col].apply(
        lambda x: find_players_in_comment(x, alias_to_player, long_aliases) if pd.notna(x) else {}
    )
    
    player_comments = df[df["players_mentioned"].apply(bool)].copy()
