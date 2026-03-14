"""ConceptNet knowledge graph reasoning tasks.

Generators that accept a NetworkX ConceptNet (sub)graph and produce
5-tuples ``(cc, query_node, target_node, answer, metadata)`` where the GNN
sees graph structure and the LLM reads concept text via ``cc.node_texts``.

Tasks (v10)
-----------
1. **kg_relation** -- Predict the relation category of a random edge (10 cls).
2. **kg_concept** -- Predict the semantic category of a masked node (9 cls).
3. **kg_pathvalid** -- Decide if a multi-hop path is valid or corrupted (2 cls).
4. **kg_analogy** -- Decide if two subgraphs are structurally analogous (3 cls).
5. **kg_cluster** -- Predict the semantic domain of a random node (6 cls).
6. **kg_transitive** -- Multi-hop transitive inference (5 cls).
7. **kg_consistency** -- Logical consistency checking (2 cls).
8. **kg_analogy_v10** -- Redesigned structural analogy with isomorphism (3 cls).
9. **kg_causal_chain** -- Causal chain coherence reasoning (3 cls).
"""

from __future__ import annotations

import random
from typing import Any

import networkx as nx
import torch

from src.data.conceptnet import (
    extract_subgraph,
    conceptnet_subgraph_to_cc,
    concept_to_text,
    categorize_relation,
    RELATION_CATEGORIES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RELATION_CLASSES: list[str] = sorted(
    {
        "IsA",
        "FormOf",
        "DerivedFrom",
        "HasContext",
        "Synonym",
        "Antonym",
        "RelatedTo",
        "UsedFor",
        "AtLocation",
        "PartOf",
        "HasSubevent",
        "CapableOf",
        "Causes",
        "HasProperty",
        "Desire",
        "Negation",
    }
)
assert len(RELATION_CLASSES) == 16

CONCEPT_CATEGORIES: list[str] = [
    "animal",
    "place",
    "activity",
    "object",
    "emotion",
    "person",
    "food",
    "body_part",
    "abstract",
]
assert len(CONCEPT_CATEGORIES) == 9

DOMAIN_CLASSES: list[str] = [
    "science",
    "everyday",
    "social",
    "spatial",
    "temporal",
    "abstract",
]
assert len(DOMAIN_CLASSES) == 6

# ---------------------------------------------------------------------------
# Keyword-based concept classifier  (task 2)
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "animal": [
        "dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep",
        "chicken", "duck", "mouse", "rat", "bear", "lion", "tiger",
        "snake", "frog", "whale", "shark", "insect", "monkey", "deer",
        "wolf", "fox", "rabbit", "turtle", "eagle", "hawk", "ant", "bee",
        "butterfly", "spider", "crab", "lobster", "squid", "octopus",
        "elephant", "giraffe", "zebra", "penguin", "dolphin", "seal",
        "bat", "owl", "parrot", "goat", "donkey", "camel", "lizard",
        "crocodile", "alligator", "gorilla", "chimpanzee", "pet",
        "mammal", "reptile", "amphibian", "vertebrate", "invertebrate",
        "creature", "beast", "organism", "species", "breed",
    ],
    "place": [
        "city", "town", "country", "park", "house", "building", "school",
        "hospital", "church", "store", "restaurant", "hotel", "beach",
        "mountain", "forest", "river", "lake", "ocean", "street", "room",
        "village", "farm", "garden", "stadium", "airport", "station",
        "museum", "library", "office", "factory", "prison", "temple",
        "mosque", "castle", "palace", "island", "desert", "valley",
        "cave", "jungle", "continent", "region", "district", "suburb",
        "neighborhood", "plaza", "harbor", "port", "mall", "theater",
        "arena", "campus", "cemetery", "zoo", "aquarium", "kitchen",
        "bedroom", "bathroom", "garage", "basement", "attic",
        "location", "area", "territory", "land", "field", "yard",
    ],
    "activity": [
        "run", "walk", "swim", "play", "dance", "sing", "cook", "drive",
        "read", "write", "climb", "jump", "throw", "catch", "fight",
        "travel", "exercise", "paint", "draw", "sleep", "fly", "ride",
        "ski", "skate", "surf", "hike", "camp", "hunt", "fish", "sail",
        "row", "dive", "box", "wrestle", "race", "compete", "train",
        "practice", "study", "learn", "teach", "work", "build", "fix",
        "clean", "wash", "sew", "knit", "garden", "farm", "mine",
        "dig", "chop", "stir", "bake", "grill", "fry", "boil",
        "shop", "buy", "sell", "trade", "invest", "gamble", "bet",
        "game", "sport", "hobby", "craft", "task", "job", "chore",
    ],
    "object": [
        "table", "chair", "car", "phone", "computer", "book", "pen",
        "door", "window", "cup", "plate", "knife", "bottle", "key",
        "clock", "lamp", "bag", "box", "tool", "wheel", "hammer",
        "nail", "screw", "rope", "chain", "wire", "pipe", "tube",
        "mirror", "pillow", "blanket", "towel", "soap", "brush",
        "comb", "scissors", "needle", "thread", "button", "zipper",
        "umbrella", "candle", "match", "battery", "camera", "radio",
        "television", "screen", "keyboard", "printer", "machine",
        "engine", "motor", "device", "instrument", "gadget", "toy",
        "ball", "doll", "puzzle", "card", "coin", "ring", "necklace",
        "hat", "shoe", "glove", "belt", "wallet", "purse", "basket",
        "bucket", "shovel", "axe", "saw", "drill", "wrench",
        "thing", "item", "stuff", "material", "equipment", "furniture",
    ],
    "emotion": [
        "love", "hate", "fear", "anger", "joy", "sadness", "happiness",
        "surprise", "disgust", "anxiety", "pride", "shame", "guilt",
        "trust", "loyalty", "envy", "jealousy", "hope", "grief",
        "excitement", "boredom", "loneliness", "frustration", "relief",
        "gratitude", "compassion", "sympathy", "empathy", "pity",
        "contempt", "awe", "wonder", "curiosity", "desire", "lust",
        "passion", "rage", "fury", "terror", "horror", "panic",
        "calm", "peace", "content", "delight", "pleasure", "bliss",
        "sorrow", "despair", "regret", "remorse", "nostalgia",
        "happy", "sad", "angry", "afraid", "scared", "worried",
        "nervous", "anxious", "excited", "bored", "lonely", "proud",
        "ashamed", "guilty", "jealous", "grateful", "confused",
        "feeling", "mood", "sentiment", "emotion", "affection",
    ],
    "person": [
        "man", "woman", "child", "baby", "friend", "teacher", "doctor",
        "father", "mother", "brother", "sister", "king", "queen",
        "soldier", "worker", "student", "artist", "leader", "hero",
        "stranger", "nurse", "lawyer", "judge", "priest", "monk",
        "farmer", "sailor", "pilot", "driver", "chef", "waiter",
        "clerk", "manager", "boss", "employee", "servant", "slave",
        "thief", "criminal", "police", "guard", "spy", "knight",
        "prince", "princess", "emperor", "president", "mayor",
        "citizen", "immigrant", "refugee", "tourist", "guest",
        "neighbor", "cousin", "uncle", "aunt", "grandfather",
        "grandmother", "husband", "wife", "son", "daughter",
        "boy", "girl", "teenager", "adult", "elder", "infant",
        "human", "people", "person", "individual", "somebody",
    ],
    "food": [
        "bread", "meat", "fruit", "vegetable", "rice", "cake", "cheese",
        "milk", "egg", "soup", "salad", "pizza", "pasta", "candy",
        "chocolate", "sugar", "salt", "butter", "apple", "banana",
        "orange", "grape", "strawberry", "lemon", "tomato", "potato",
        "onion", "carrot", "corn", "bean", "pea", "nut", "seed",
        "wheat", "flour", "dough", "sauce", "spice", "pepper", "garlic",
        "ginger", "honey", "jam", "jelly", "cream", "yogurt", "ice",
        "pie", "cookie", "biscuit", "muffin", "donut", "pancake",
        "waffle", "cereal", "oatmeal", "noodle", "dumpling", "sushi",
        "steak", "bacon", "sausage", "ham", "turkey", "lobster",
        "shrimp", "oyster", "clam", "tuna", "salmon", "cod",
        "snack", "meal", "dish", "recipe", "ingredient", "cuisine",
        "breakfast", "lunch", "dinner", "dessert", "appetizer",
    ],
    "body_part": [
        "head", "hand", "foot", "eye", "ear", "nose", "mouth", "arm",
        "leg", "finger", "toe", "heart", "brain", "bone", "skin",
        "hair", "tooth", "tongue", "knee", "shoulder", "elbow",
        "wrist", "ankle", "hip", "neck", "chest", "back", "stomach",
        "liver", "kidney", "lung", "muscle", "nerve", "vein",
        "artery", "blood", "spine", "skull", "rib", "jaw", "chin",
        "cheek", "forehead", "eyebrow", "eyelash", "lip", "throat",
        "palm", "thumb", "nail", "heel", "calf", "thigh", "waist",
        "belly", "abdomen", "pelvis", "organ", "tissue", "limb",
    ],
    "abstract": [
        "time", "space", "idea", "thought", "mind", "soul", "truth",
        "knowledge", "science", "math", "philosophy", "freedom",
        "justice", "power", "energy", "force", "law", "theory",
        "concept", "reason", "logic", "wisdom", "intelligence",
        "consciousness", "memory", "dream", "imagination", "belief",
        "faith", "religion", "spirit", "god", "heaven", "hell",
        "death", "life", "birth", "fate", "destiny", "luck", "chance",
        "risk", "value", "meaning", "purpose", "goal", "plan",
        "strategy", "method", "process", "system", "structure",
        "pattern", "rule", "principle", "right", "wrong", "good",
        "evil", "virtue", "sin", "duty", "honor", "dignity",
    ],
}

# IsA hypernyms that map to concept categories
_ISA_CATEGORY_MAP: dict[str, str] = {
    "animal": "animal", "mammal": "animal", "bird": "animal",
    "fish": "animal", "insect": "animal", "reptile": "animal",
    "amphibian": "animal", "vertebrate": "animal", "invertebrate": "animal",
    "creature": "animal", "organism": "animal", "pet": "animal",
    "place": "place", "location": "place", "city": "place",
    "country": "place", "building": "place", "room": "place",
    "region": "place", "area": "place", "territory": "place",
    "activity": "activity", "sport": "activity", "game": "activity",
    "exercise": "activity", "hobby": "activity", "action": "activity",
    "event": "activity", "task": "activity", "work": "activity",
    "object": "object", "tool": "object", "device": "object",
    "instrument": "object", "machine": "object", "equipment": "object",
    "furniture": "object", "vehicle": "object", "weapon": "object",
    "container": "object", "appliance": "object", "artifact": "object",
    "thing": "object", "item": "object", "substance": "object",
    "emotion": "emotion", "feeling": "emotion", "mood": "emotion",
    "sentiment": "emotion", "sensation": "emotion", "affect": "emotion",
    "person": "person", "human": "person", "people": "person",
    "man": "person", "woman": "person", "child": "person",
    "professional": "person", "worker": "person", "leader": "person",
    "food": "food", "fruit": "food", "vegetable": "food",
    "meat": "food", "drink": "food", "beverage": "food",
    "meal": "food", "dish": "food", "snack": "food", "ingredient": "food",
    "body_part": "body_part", "organ": "body_part", "limb": "body_part",
    "muscle": "body_part", "bone": "body_part", "tissue": "body_part",
}


def _classify_via_isa(concept: str, G: nx.Graph | None,
                      category_map: dict[str, str]) -> str | None:
    """Follow IsA edges in ConceptNet to classify a concept."""
    if G is None or concept not in G:
        return None
    # Check 1-hop and 2-hop IsA targets
    for neighbor in G.neighbors(concept):
        edge_data = G.edges[concept, neighbor]
        rel = edge_data.get("relation", "")
        if rel in ("IsA", "DerivedFrom", "InstanceOf"):
            target_text = concept_to_text(neighbor).lower()
            for token in target_text.split():
                if token in category_map:
                    return category_map[token]
            # 2nd hop
            if neighbor in G:
                for n2 in G.neighbors(neighbor):
                    edge2 = G.edges[neighbor, n2]
                    if edge2.get("relation", "") in ("IsA", "DerivedFrom"):
                        t2 = concept_to_text(n2).lower()
                        for token in t2.split():
                            if token in category_map:
                                return category_map[token]
    return None


def classify_concept(concept: str, G: nx.Graph | None = None) -> str:
    """Classify a concept string into one of 9 categories.

    Uses ConceptNet IsA hierarchy (if *G* provided), then keyword matching,
    then uniform random fallback (not always "abstract").

    Args:
        concept: A concept slug like ``"dog"`` or ``"hot_dog"``.
        G: Optional ConceptNet graph for IsA-based classification.

    Returns:
        Category string from :data:`CONCEPT_CATEGORIES`.
    """
    # Strategy 1: ConceptNet IsA hierarchy
    isa_result = _classify_via_isa(concept, G, _ISA_CATEGORY_MAP)
    if isa_result is not None:
        return isa_result

    # Strategy 2: keyword matching on concept text
    text = concept_to_text(concept).lower()
    tokens = set(text.split())

    best_cat = None
    best_score = 0

    for cat, keywords in _CATEGORY_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in tokens or kw in text:
                score += 1
        if score > best_score:
            best_score = score
            best_cat = cat

    if best_cat is not None:
        return best_cat

    # Strategy 3: uniform random fallback (avoids class collapse)
    return random.choice(CONCEPT_CATEGORIES)


# ---------------------------------------------------------------------------
# Keyword-based domain classifier  (task 5)
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "science": [
        "atom", "molecule", "electron", "proton", "neutron", "cell",
        "gene", "protein", "enzyme", "photosynthesis", "gravity",
        "energy", "radiation", "spectrum", "experiment", "chemical",
        "element", "compound", "reaction", "formula", "equation",
        "hypothesis", "laboratory", "microscope", "telescope",
        "biology", "chemistry", "physics", "geology", "astronomy",
        "mathematics", "calculus", "algebra", "geometry", "statistic",
        "virus", "bacteria", "dna", "rna", "chromosome", "mutation",
        "evolution", "fossil", "dinosaur", "mineral", "crystal",
        "magnet", "electric", "circuit", "voltage", "frequency",
        "wavelength", "laser", "nuclear", "quantum", "relativity",
        "temperature", "pressure", "density", "velocity", "mass",
        "volume", "orbit", "planet", "star", "galaxy", "comet",
        "oxygen", "hydrogen", "carbon", "nitrogen", "helium",
        "acid", "base", "solution", "catalyst", "polymer",
        "research", "discovery", "theorem", "proof", "variable",
    ],
    "everyday": [
        "house", "car", "food", "cook", "clean", "shop", "work",
        "sleep", "eat", "drink", "walk", "phone", "clothes", "money",
        "home", "door", "window", "kitchen", "bedroom", "bathroom",
        "table", "chair", "bed", "sofa", "lamp", "cup", "plate",
        "fork", "spoon", "knife", "towel", "soap", "brush", "comb",
        "shirt", "pants", "shoe", "hat", "jacket", "dress", "sock",
        "bag", "wallet", "key", "lock", "umbrella", "basket",
        "wash", "dry", "iron", "fold", "sweep", "mop", "vacuum",
        "grocery", "laundry", "trash", "garbage", "recycle",
        "pet", "dog", "cat", "plant", "flower", "garden", "lawn",
        "breakfast", "lunch", "dinner", "snack", "coffee", "tea",
        "bread", "butter", "cheese", "milk", "egg", "rice", "pasta",
        "fruit", "vegetable", "meat", "fish", "chicken", "sugar",
        "drive", "park", "gas", "tire", "repair", "bill", "rent",
        "pay", "price", "cost", "cheap", "expensive", "sale",
    ],
    "social": [
        "friend", "family", "love", "trust", "help", "community",
        "leader", "team", "party", "wedding", "meeting", "argue",
        "agree", "share", "teach", "parent", "child", "brother",
        "sister", "husband", "wife", "neighbor", "colleague",
        "group", "club", "organization", "society", "culture",
        "tradition", "custom", "ceremony", "celebration", "festival",
        "church", "temple", "mosque", "religion", "faith", "prayer",
        "government", "politics", "vote", "election", "democracy",
        "law", "court", "judge", "police", "crime", "punishment",
        "war", "peace", "conflict", "negotiate", "cooperate",
        "communicate", "speak", "listen", "discuss", "debate",
        "marry", "divorce", "birth", "funeral", "greet", "thank",
        "apologize", "forgive", "promise", "betray", "respect",
        "power", "authority", "status", "class", "equality", "rights",
        "education", "school", "university", "student", "teacher",
    ],
    "spatial": [
        "city", "park", "mountain", "river", "ocean", "forest",
        "street", "room", "building", "bridge", "road", "map",
        "north", "south", "location", "east", "west", "left", "right",
        "up", "down", "above", "below", "inside", "outside", "near",
        "far", "close", "distant", "deep", "shallow", "high", "low",
        "wide", "narrow", "long", "short", "big", "small", "large",
        "tiny", "huge", "tall", "flat", "round", "square", "straight",
        "curved", "corner", "edge", "center", "middle", "border",
        "surface", "ground", "floor", "ceiling", "wall", "roof",
        "country", "continent", "island", "desert", "valley", "hill",
        "lake", "sea", "beach", "coast", "shore", "harbor", "port",
        "village", "town", "suburb", "district", "region", "area",
        "field", "farm", "yard", "path", "trail", "highway", "tunnel",
        "direction", "distance", "position", "place", "space",
    ],
    "temporal": [
        "morning", "night", "year", "season", "winter", "summer",
        "spring", "autumn", "hour", "minute", "yesterday", "tomorrow",
        "clock", "calendar", "history", "day", "week", "month",
        "decade", "century", "millennium", "second", "noon", "midnight",
        "dawn", "dusk", "sunset", "sunrise", "evening", "afternoon",
        "today", "tonight", "weekend", "weekday", "holiday", "birthday",
        "anniversary", "deadline", "schedule", "appointment", "date",
        "age", "era", "period", "epoch", "ancient", "modern",
        "past", "present", "future", "begin", "end", "start", "finish",
        "early", "late", "soon", "never", "always", "often", "rarely",
        "before", "after", "during", "while", "until", "since",
        "wait", "hurry", "rush", "delay", "pause", "continue",
        "old", "young", "new", "recent", "current", "previous", "next",
    ],
    "abstract": [
        "idea", "thought", "concept", "theory", "truth", "freedom",
        "justice", "knowledge", "mind", "reason", "logic", "philosophy",
        "meaning", "purpose", "belief", "wisdom", "intelligence",
        "consciousness", "dream", "imagination", "creativity",
        "beauty", "art", "music", "poetry", "literature", "story",
        "language", "word", "number", "symbol", "sign", "code",
        "information", "data", "pattern", "structure", "system",
        "method", "process", "strategy", "plan", "goal", "value",
        "principle", "rule", "standard", "quality", "quantity",
        "cause", "effect", "result", "consequence", "problem",
        "solution", "question", "answer", "argument", "evidence",
        "fact", "opinion", "assumption", "conclusion", "definition",
    ],
}

# IsA hypernyms that map to domain classes
_ISA_DOMAIN_MAP: dict[str, str] = {
    "science": "science", "biology": "science", "chemistry": "science",
    "physics": "science", "mathematics": "science", "medicine": "science",
    "technology": "science", "engineering": "science", "research": "science",
    "experiment": "science", "chemical": "science", "element": "science",
    "everyday": "everyday", "household": "everyday", "domestic": "everyday",
    "routine": "everyday", "chore": "everyday", "appliance": "everyday",
    "furniture": "everyday", "clothing": "everyday", "food": "everyday",
    "tool": "everyday", "utensil": "everyday", "vehicle": "everyday",
    "social": "social", "relationship": "social", "community": "social",
    "organization": "social", "institution": "social", "ceremony": "social",
    "communication": "social", "interaction": "social", "society": "social",
    "culture": "social", "religion": "social", "politics": "social",
    "government": "social", "education": "social", "profession": "social",
    "spatial": "spatial", "location": "spatial", "place": "spatial",
    "region": "spatial", "area": "spatial", "direction": "spatial",
    "distance": "spatial", "position": "spatial", "geography": "spatial",
    "landscape": "spatial", "terrain": "spatial", "building": "spatial",
    "temporal": "temporal", "time": "temporal", "period": "temporal",
    "duration": "temporal", "season": "temporal", "era": "temporal",
    "schedule": "temporal", "event": "temporal", "occasion": "temporal",
    "abstract": "abstract", "concept": "abstract", "idea": "abstract",
    "theory": "abstract", "philosophy": "abstract", "logic": "abstract",
    "principle": "abstract", "quality": "abstract", "property": "abstract",
}


def classify_domain(concept: str, G: nx.Graph | None = None) -> str:
    """Classify a concept string into one of 6 semantic domains.

    Uses ConceptNet IsA hierarchy (if *G* provided), then keyword matching,
    then uniform random fallback (not always "abstract").

    Args:
        concept: A concept slug like ``"atom"`` or ``"hot_dog"``.
        G: Optional ConceptNet graph for IsA-based classification.

    Returns:
        Domain string from :data:`DOMAIN_CLASSES`.
    """
    # Strategy 1: ConceptNet IsA hierarchy
    isa_result = _classify_via_isa(concept, G, _ISA_DOMAIN_MAP)
    if isa_result is not None:
        return isa_result

    # Strategy 2: keyword matching
    text = concept_to_text(concept).lower()
    tokens = set(text.split())

    best_domain = None
    best_score = 0

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in tokens or kw in text:
                score += 1
        if score > best_score:
            best_score = score
            best_domain = domain

    if best_domain is not None:
        return best_domain

    # Strategy 3: uniform random fallback
    return random.choice(DOMAIN_CLASSES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_RETRIES = 10


def _safe_extract(
    G: nx.Graph,
    min_nodes: int,
    max_nodes: int,
) -> tuple[nx.Graph, list[str]]:
    """Try to extract a valid subgraph, retrying on failure."""
    last_err: Exception | None = None
    for _ in range(_MAX_RETRIES):
        try:
            sub, node_list = extract_subgraph(
                G, seed=None, min_nodes=min_nodes, max_nodes=max_nodes,
            )
            if sub.number_of_nodes() >= min_nodes:
                return sub, node_list
        except (ValueError, IndexError) as exc:
            last_err = exc
            continue

    # Final fallback: just use whatever the graph gives us
    try:
        sub, node_list = extract_subgraph(
            G, seed=None, min_nodes=1, max_nodes=max_nodes,
        )
        return sub, node_list
    except (ValueError, IndexError):
        pass

    if last_err:
        raise last_err
    raise ValueError(f"Cannot extract subgraph from graph with {G.number_of_nodes()} nodes")


def _build_node_texts(sub: nx.Graph, node_list: list[str]) -> dict[str, str]:
    """Map each concept in *node_list* to readable text."""
    return {c: concept_to_text(c) for c in node_list if c in sub}


# ---------------------------------------------------------------------------
# Task 1: Relation classification
# ---------------------------------------------------------------------------


def _get_relation_index(G: nx.Graph) -> dict[str, list[tuple[str, str]]]:
    """Build category → [(u, v), ...] index for balanced relation sampling.

    Cached on the graph object as ``_relation_index`` to avoid re-scanning.
    """
    if hasattr(G, '_relation_index'):
        return G._relation_index  # type: ignore[attr-defined]

    index: dict[str, list[tuple[str, str]]] = {cat: [] for cat in RELATION_CLASSES}
    for u, v, data in G.edges(data=True):
        raw_rel = data.get("relation", data.get("raw_relation", "RelatedTo"))
        cat = categorize_relation(raw_rel)
        if cat in index:
            index[cat].append((u, v))

    # Remove empty categories
    index = {k: v for k, v in index.items() if v}

    G._relation_index = index  # type: ignore[attr-defined]
    return index


def generate_kg_relation_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 20,
    max_nodes: int = 50,
) -> tuple:
    """Predict the relation category of a random edge.

    Uses balanced sampling: picks a target relation category uniformly,
    then seeds the subgraph around an edge of that type. This prevents
    ConceptNet's natural bias (86%+ RelatedTo) from dominating the dataset.

    Args:
        G: Full (or partial) ConceptNet graph.
        embedding_dim: CellComplex embedding dimension (>= 4).
        min_nodes: Minimum subgraph size.
        max_nodes: Maximum subgraph size.

    Returns:
        ``(cc, query_node_idx, target_node_idx, answer, metadata)`` where
        *answer* is an index into :data:`RELATION_CLASSES` (0--9).
    """
    rel_index = _get_relation_index(G)
    available_cats = list(rel_index.keys())
    if not available_cats:
        raise ValueError("No relation categories found in graph")

    # Pick a target category uniformly from available categories
    target_category = random.choice(available_cats)

    # Pick a random edge of that category from the full graph
    u_seed, v_seed = random.choice(rel_index[target_category])

    # Build subgraph seeded from one of the edge's endpoints
    seed_node = random.choice([u_seed, v_seed])
    u, v = None, None
    for _ in range(_MAX_RETRIES):
        try:
            sub, node_list = extract_subgraph(
                G, seed=seed_node, min_nodes=min_nodes, max_nodes=max_nodes,
            )
        except (ValueError, IndexError):
            # Seed node might have too few neighbors — pick a new edge
            u_seed, v_seed = random.choice(rel_index[target_category])
            seed_node = random.choice([u_seed, v_seed])
            continue

        # Find an edge of the target category in this subgraph
        matching = [
            (eu, ev, d) for eu, ev, d in sub.edges(data=True)
            if categorize_relation(d.get("relation", d.get("raw_relation", "RelatedTo"))) == target_category
        ]
        if matching:
            u, v, data = random.choice(matching)
            break
        # Subgraph didn't contain target category — retry with new seed edge
        u_seed, v_seed = random.choice(rel_index[target_category])
        seed_node = random.choice([u_seed, v_seed])

    # Fallback: if we never found a match, use _safe_extract with any edge
    if u is None:
        sub, node_list = _safe_extract(G, min_nodes, max_nodes)
        edges = list(sub.edges(data=True))
        if not edges:
            raise ValueError("Subgraph has no edges")
        u, v, data = random.choice(edges)

    raw_rel = data.get("relation", data.get("raw_relation", "RelatedTo"))
    category = categorize_relation(raw_rel)
    answer = RELATION_CLASSES.index(category) if category in RELATION_CLASSES else RELATION_CLASSES.index("RelatedTo")

    cc, node_map, _ = conceptnet_subgraph_to_cc(sub, embedding_dim, node_list)

    # Mask the target edge's relation encoding to prevent data leakage
    # (the model shouldn't be able to read the answer from the edge embedding)
    from src.data.conceptnet import RELATION_CATEGORIES
    n_rels = len(RELATION_CATEGORIES)
    qi, ti = node_map[u], node_map[v]
    B1 = cc.boundary_operator(1)
    if B1 is not None and B1.numel() > 0:
        edge_embs = cc.get_embeddings(1)
        for edge_idx in range(B1.shape[1]):
            col = B1[:, edge_idx]
            nz = col.nonzero(as_tuple=False).squeeze(-1).tolist()
            if set(nz) == {qi, ti}:
                edge_embs[edge_idx, 1:min(1 + n_rels, edge_embs.shape[1])] = 0.0
                break
        cc.set_embeddings(1, edge_embs)

    node_texts = _build_node_texts(sub, node_list)

    query_idx = node_map[u]
    target_idx = node_map[v]

    metadata = {
        "task_type": "kg_relation",
        "node_texts": node_texts,
        "task_prompt": (
            f"Predict the relation between '{concept_to_text(u)}' and "
            f"'{concept_to_text(v)}' | "
            f"nodes={sub.number_of_nodes()} edges={sub.number_of_edges()} | "
            f"task=kg_relation"
        ),
        "relation": category,
        "num_classes": len(RELATION_CLASSES),
    }

    return cc, query_idx, target_idx, answer, metadata


# ---------------------------------------------------------------------------
# Task 2: Concept category classification (masked)
# ---------------------------------------------------------------------------


def generate_kg_concept_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 20,
    max_nodes: int = 50,
) -> tuple:
    """Predict the category of a masked concept node.

    One node's text is replaced with ``"[MASK]"`` in ``cc.node_texts``.
    The model must infer the semantic category from the graph neighbourhood.

    Args:
        G: Full (or partial) ConceptNet graph.
        embedding_dim: CellComplex embedding dimension (>= 4).
        min_nodes: Minimum subgraph size.
        max_nodes: Maximum subgraph size.

    Returns:
        ``(cc, query_node_idx, target_node_idx, answer, metadata)`` where
        *answer* is an index into :data:`CONCEPT_CATEGORIES` (0--8).
        *target_node_idx* is set to a random neighbor of the masked node.
    """
    sub, node_list = _safe_extract(G, min_nodes, max_nodes)

    # Pick a node to mask
    masked_concept = random.choice(node_list)
    category = classify_concept(masked_concept, G)
    answer = CONCEPT_CATEGORIES.index(category)

    cc, node_map, _ = conceptnet_subgraph_to_cc(sub, embedding_dim, node_list)

    # Replace the masked node's text in cc.node_texts
    masked_idx = node_map[masked_concept]
    cc.node_texts[masked_idx] = "[MASK]"

    # Pick a neighbor as target node (gives the model spatial context)
    neighbors = list(sub.neighbors(masked_concept))
    if neighbors:
        target_concept = random.choice(neighbors)
    else:
        # Pick any other node
        others = [c for c in node_list if c != masked_concept and c in node_map]
        target_concept = random.choice(others) if others else masked_concept

    target_idx = node_map[target_concept]

    node_texts = _build_node_texts(sub, node_list)
    node_texts[masked_concept] = "[MASK]"

    metadata = {
        "task_type": "kg_concept",
        "node_texts": node_texts,
        "task_prompt": (
            f"Predict the category of the masked concept at node {masked_idx} | "
            f"neighbors={len(neighbors)} | "
            f"nodes={sub.number_of_nodes()} edges={sub.number_of_edges()} | "
            f"task=kg_concept"
        ),
        "masked_concept": masked_concept,
        "true_category": category,
        "num_classes": len(CONCEPT_CATEGORIES),
    }

    return cc, masked_idx, target_idx, answer, metadata


# ---------------------------------------------------------------------------
# Task 3: Path validity
# ---------------------------------------------------------------------------


def generate_kg_pathvalid_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 20,
    max_nodes: int = 50,
) -> tuple:
    """Decide if a multi-hop path in the knowledge graph is valid or corrupted.

    Finds a shortest path of 3--5 hops. With 50 % probability the path is
    kept valid (answer = 1); otherwise the middle node is swapped with a
    random same-degree node (answer = 0).

    Args:
        G: Full (or partial) ConceptNet graph.
        embedding_dim: CellComplex embedding dimension (>= 4).
        min_nodes: Minimum subgraph size.
        max_nodes: Maximum subgraph size.

    Returns:
        ``(cc, query_node_idx, target_node_idx, answer, metadata)`` where
        *answer* is 0 (corrupted) or 1 (valid).
    """
    sub, node_list = _safe_extract(G, min_nodes, max_nodes)
    nodes = list(sub.nodes())

    # Find a path of 3-5 hops
    path = None
    for _ in range(50):
        src, tgt = random.sample(nodes, 2)
        try:
            candidate = nx.shortest_path(sub, src, tgt)
        except nx.NetworkXNoPath:
            continue
        if 3 <= len(candidate) - 1 <= 5:
            path = candidate
            break

    if path is None:
        # Fallback: take any shortest path >= 2 hops
        for _ in range(50):
            src, tgt = random.sample(nodes, 2)
            try:
                candidate = nx.shortest_path(sub, src, tgt)
            except nx.NetworkXNoPath:
                continue
            if len(candidate) >= 3:
                path = candidate
                break

    if path is None:
        # Ultimate fallback: pick two connected nodes + a neighbor
        src = random.choice(nodes)
        neighbors = list(sub.neighbors(src))
        if neighbors:
            tgt = random.choice(neighbors)
            path = [src, tgt]
        else:
            path = [src, nodes[(nodes.index(src) + 1) % len(nodes)]]

    src_concept = path[0]
    tgt_concept = path[-1]

    # 50% corrupt, 50% valid
    if random.random() < 0.5 and len(path) >= 3:
        # Corrupt: swap a middle node AND rewire edges in the graph
        mid_idx = len(path) // 2
        mid_node = path[mid_idx]
        mid_degree = sub.degree(mid_node)
        prev_node = path[mid_idx - 1]
        next_node = path[mid_idx + 1]

        # Find candidates with similar degree (within +/- 2)
        candidates = [
            n for n in nodes
            if n not in path and abs(sub.degree(n) - mid_degree) <= 2
        ]
        if not candidates:
            candidates = [n for n in nodes if n not in path]

        if candidates:
            swap_node = random.choice(candidates)
            path[mid_idx] = swap_node

            # Rewire the graph: remove path edges through mid, add through swap
            sub = sub.copy()  # don't mutate the original
            if sub.has_edge(prev_node, mid_node):
                sub.remove_edge(prev_node, mid_node)
            if sub.has_edge(mid_node, next_node):
                sub.remove_edge(mid_node, next_node)
            # Add edges through the swap node (creates structural discontinuity)
            sub.add_edge(prev_node, swap_node)
            sub.add_edge(swap_node, next_node)

            # Update node_list if swap_node wasn't already in subgraph
            if swap_node not in node_list:
                node_list = list(node_list) + [swap_node]

            answer = 0
        else:
            answer = 1  # can't corrupt, keep valid
    else:
        answer = 1

    cc, node_map, _ = conceptnet_subgraph_to_cc(sub, embedding_dim, node_list)
    node_texts = _build_node_texts(sub, node_list)

    query_idx = node_map[src_concept]
    target_idx = node_map[tgt_concept]

    path_text = " -> ".join(concept_to_text(c) for c in path)

    metadata = {
        "task_type": "kg_pathvalid",
        "node_texts": node_texts,
        "task_prompt": (
            f"Is the path valid? {path_text} | "
            f"hops={len(path) - 1} | "
            f"nodes={sub.number_of_nodes()} edges={sub.number_of_edges()} | "
            f"task=kg_pathvalid"
        ),
        "path": [concept_to_text(c) for c in path],
        "path_length": len(path) - 1,
        "num_classes": 2,
    }

    return cc, query_idx, target_idx, answer, metadata


# ---------------------------------------------------------------------------
# Task 4: Structural analogy
# ---------------------------------------------------------------------------


def generate_kg_analogy_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 10,
    max_nodes: int = 30,
) -> tuple:
    """Decide if two knowledge subgraphs are structurally analogous.

    Extracts two subgraphs from different seed nodes, computes the
    relation-type overlap (Jaccard), and classifies:
      - overlap > 0.6 -> analogous (2)
      - overlap > 0.3 -> partial (1)
      - else           -> not analogous (0)

    Both subgraphs are merged into one CellComplex with prefixed node names.

    Args:
        G: Full (or partial) ConceptNet graph.
        embedding_dim: CellComplex embedding dimension (>= 4).
        min_nodes: Minimum per-subgraph size.
        max_nodes: Maximum per-subgraph size.

    Returns:
        ``(cc, query_node_idx, target_node_idx, answer, metadata)`` where
        *answer* is 0 (not analogous), 1 (partial), or 2 (analogous).
    """
    from src.cell_complex.cell_complex import CellComplex as CC
    from src.benchmarks.topological_tasks import auto_fill_triangles

    # Extract two subgraphs from different seeds
    sub_a, nlist_a = _safe_extract(G, min_nodes, max_nodes)
    sub_b, nlist_b = _safe_extract(G, min_nodes, max_nodes)

    # Compute relation overlap (Jaccard on the multiset of relation types)
    rels_a = set(
        data.get("relation", "RelatedTo") for _, _, data in sub_a.edges(data=True)
    )
    rels_b = set(
        data.get("relation", "RelatedTo") for _, _, data in sub_b.edges(data=True)
    )

    if rels_a or rels_b:
        overlap = len(rels_a & rels_b) / len(rels_a | rels_b)
    else:
        overlap = 0.0

    if overlap > 0.6:
        answer = 2  # analogous
    elif overlap > 0.3:
        answer = 1  # partial
    else:
        answer = 0  # not analogous

    # Build a merged graph with prefixed node names
    merged = nx.Graph()

    # Precompute structural features for sub_a
    degrees_a = dict(sub_a.degree())
    max_deg_a = max(degrees_a.values()) if degrees_a else 1
    max_deg_a = max(max_deg_a, 1)
    clust_a = nx.clustering(sub_a)

    # Precompute structural features for sub_b
    degrees_b = dict(sub_b.degree())
    max_deg_b = max(degrees_b.values()) if degrees_b else 1
    max_deg_b = max(max_deg_b, 1)
    clust_b = nx.clustering(sub_b)

    for node in sub_a.nodes():
        merged.add_node(f"a_{node}", prefix="a", original=node)
    for u, v, data in sub_a.edges(data=True):
        merged.add_edge(f"a_{u}", f"a_{v}", **data)

    for node in sub_b.nodes():
        merged.add_node(f"b_{node}", prefix="b", original=node)
    for u, v, data in sub_b.edges(data=True):
        merged.add_edge(f"b_{u}", f"b_{v}", **data)

    # Build CellComplex from the merged graph
    cc = CC(embedding_dim=embedding_dim)
    node_map: dict[str, int] = {}
    merged_nodes = list(merged.nodes())
    merged_degrees = dict(merged.degree())
    merged_max_deg = max(merged_degrees.values()) if merged_degrees else 1
    merged_max_deg = max(merged_max_deg, 1)
    merged_clust = nx.clustering(merged)

    for mnode in merged_nodes:
        emb = torch.randn(embedding_dim) * 0.01
        emb[0] = merged_degrees[mnode] / merged_max_deg
        emb[1] = merged_clust[mnode]
        emb[2] = 0.0  # no BFS for merged graph

        cc_idx = cc.add_0_cell(emb, "node")
        node_map[mnode] = cc_idx

    # Populate node_texts using the original concept names
    cc.node_texts = []
    for mnode in merged_nodes:
        original = merged.nodes[mnode].get("original", mnode)
        prefix = merged.nodes[mnode].get("prefix", "")
        cc.node_texts.append(f"{prefix}: {concept_to_text(original)}")

    # Add edges with relation-type encoding
    from src.data.conceptnet import RELATION_CATEGORIES
    rel_to_idx = {r: i for i, r in enumerate(RELATION_CATEGORIES)}
    n_rels = len(RELATION_CATEGORIES)

    for u, v, data in merged.edges(data=True):
        emb = torch.randn(embedding_dim) * 0.01
        weight = data.get("weight", 1.0)
        relation = data.get("relation", "RelatedTo")
        emb[0] = weight / 10.0
        # One-hot relation type in dims 1..n_rels
        rel_idx = rel_to_idx.get(relation, rel_to_idx.get("RelatedTo", 0))
        if 1 + rel_idx < embedding_dim:
            emb[1:min(1 + n_rels, embedding_dim)] = 0.0
            emb[1 + rel_idx] = 1.0
        cc.add_1_cell(node_map[u], node_map[v], emb, relation)

    auto_fill_triangles(cc)

    # Pick representative nodes from each subgraph
    query_node = f"a_{nlist_a[0]}"
    target_node = f"b_{nlist_b[0]}"
    query_idx = node_map[query_node]
    target_idx = node_map[target_node]

    node_texts_a = {c: concept_to_text(c) for c in nlist_a if c in sub_a}
    node_texts_b = {c: concept_to_text(c) for c in nlist_b if c in sub_b}
    combined_texts = {}
    for k, v_ in node_texts_a.items():
        combined_texts[f"a_{k}"] = v_
    for k, v_ in node_texts_b.items():
        combined_texts[f"b_{k}"] = v_

    metadata = {
        "task_type": "kg_analogy",
        "node_texts": combined_texts,
        "task_prompt": (
            f"Are these subgraphs analogous? overlap={overlap:.3f} | "
            f"rels_a={sorted(rels_a)} rels_b={sorted(rels_b)} | "
            f"nodes_a={sub_a.number_of_nodes()} nodes_b={sub_b.number_of_nodes()} | "
            f"task=kg_analogy"
        ),
        "overlap": overlap,
        "rels_a": sorted(rels_a),
        "rels_b": sorted(rels_b),
        "num_classes": 3,
    }

    return cc, query_idx, target_idx, answer, metadata


# ---------------------------------------------------------------------------
# Task 5: Semantic domain clustering
# ---------------------------------------------------------------------------


def generate_kg_cluster_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 20,
    max_nodes: int = 50,
) -> tuple:
    """Predict the semantic domain of a random node.

    Args:
        G: Full (or partial) ConceptNet graph.
        embedding_dim: CellComplex embedding dimension (>= 4).
        min_nodes: Minimum subgraph size.
        max_nodes: Maximum subgraph size.

    Returns:
        ``(cc, query_node_idx, target_node_idx, answer, metadata)`` where
        *answer* is an index into :data:`DOMAIN_CLASSES` (0--5).
    """
    sub, node_list = _safe_extract(G, min_nodes, max_nodes)

    # Pick a random node
    query_concept = random.choice(node_list)
    domain = classify_domain(query_concept, G)
    answer = DOMAIN_CLASSES.index(domain)

    cc, node_map, _ = conceptnet_subgraph_to_cc(sub, embedding_dim, node_list)
    node_texts = _build_node_texts(sub, node_list)

    query_idx = node_map[query_concept]

    # Pick a neighbor as target for spatial context
    neighbors = list(sub.neighbors(query_concept))
    if neighbors:
        target_concept = random.choice(neighbors)
    else:
        others = [c for c in node_list if c != query_concept and c in node_map]
        target_concept = random.choice(others) if others else query_concept
    target_idx = node_map[target_concept]

    metadata = {
        "task_type": "kg_cluster",
        "node_texts": node_texts,
        "task_prompt": (
            f"Predict the semantic domain of '{concept_to_text(query_concept)}' | "
            f"neighbors={len(neighbors)} | "
            f"nodes={sub.number_of_nodes()} edges={sub.number_of_edges()} | "
            f"task=kg_cluster"
        ),
        "query_concept": query_concept,
        "true_domain": domain,
        "num_classes": len(DOMAIN_CLASSES),
    }

    return cc, query_idx, target_idx, answer, metadata


# ---------------------------------------------------------------------------
# v10 Task 6: Transitive inference
# ---------------------------------------------------------------------------

TRANSITIVE_CLASSES: list[str] = ["Causes", "HasA", "IsA", "PartOf", "none"]
assert len(TRANSITIVE_CLASSES) == 5

_TRANSITIVE_RELATIONS = {"IsA", "PartOf", "HasA"}


def generate_kg_transitive_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 20,
    max_nodes: int = 50,
) -> tuple | None:
    """Generate transitive inference task with balanced class sampling.

    Picks a target class first (uniform over TRANSITIVE_CLASSES), then
    searches for a matching 2-hop chain. This prevents "none" from
    dominating the dataset.
    """
    target_class = random.choice(TRANSITIVE_CLASSES)

    for _ in range(_MAX_RETRIES):
        sub, node_list = _safe_extract(G, min_nodes, max_nodes)

        # Collect all 2-hop chains and classify them
        chains: list[tuple[str, str, str, str, str, str]] = []  # (a, b, c, rel_ab, rel_bc, class)
        nodes = list(sub.nodes())
        random.shuffle(nodes)
        for a in nodes:
            for b in (sub.successors(a) if sub.is_directed() else sub.neighbors(a)):
                rel_ab = sub[a][b].get('relation', '')
                successors_b = sub.successors(b) if sub.is_directed() else sub.neighbors(b)
                for c in successors_b:
                    if c == a:
                        continue
                    rel_bc = sub[b][c].get('relation', '')
                    if rel_ab == rel_bc and rel_ab in _TRANSITIVE_RELATIONS:
                        cls = rel_ab if rel_ab in TRANSITIVE_CLASSES else "none"
                    else:
                        cls = "none"
                    chains.append((a, b, c, rel_ab, rel_bc, cls))

        if not chains:
            continue

        # Filter for target class
        matching = [ch for ch in chains if ch[5] == target_class]
        if not matching:
            # Accept any chain from this subgraph
            matching = chains

        a, b, c, rel_ab, rel_bc, answer_str = random.choice(matching)
        answer = TRANSITIVE_CLASSES.index(answer_str)

        cc, node_map, _ = conceptnet_subgraph_to_cc(sub, embedding_dim, node_list)
        query_idx = node_map[a]
        target_idx = node_map[c]

        node_texts = _build_node_texts(sub, node_list)

        meta = {
            'task_type': 'kg_transitive',
            'node_texts': node_texts,
            'chain': [a, b, c],
            'relations': [rel_ab, rel_bc],
            'answer_str': answer_str,
            'num_classes': len(TRANSITIVE_CLASSES),
        }
        return cc, query_idx, target_idx, answer, meta

    return None


# ---------------------------------------------------------------------------
# v10 Task 7: Logical consistency checking
# ---------------------------------------------------------------------------


def generate_kg_consistency_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 20,
    max_nodes: int = 50,
) -> tuple | None:
    """Generate logical consistency checking task.

    50% of samples are consistent (answer=1), 50% have a semantic
    contradiction (answer=0). Corruption: replace one edge's target with
    a concept from a different branch.
    """
    sub, node_list = _safe_extract(G, min_nodes, max_nodes)

    corrupted = random.random() < 0.5
    u_node = None
    replacement = None

    if corrupted:
        isa_edges = [(u, v) for u, v, d in sub.edges(data=True)
                     if d.get('relation') == 'IsA']
        if not isa_edges:
            corrupted = False
        else:
            u_node, v_node = random.choice(isa_edges)
            # Find candidates: nodes that are NOT in the IsA ancestry of u
            all_nodes = list(sub.nodes())
            candidates = [n for n in all_nodes if n != u_node and n != v_node]

            if candidates:
                replacement = random.choice(candidates)
                sub = sub.copy()
                sub.remove_edge(u_node, v_node)
                sub.add_edge(u_node, replacement, relation='IsA', weight=1.0)
                # Update node_list if replacement wasn't in it
                if replacement not in node_list:
                    node_list = list(node_list) + [replacement]
            else:
                corrupted = False

    answer = 0 if corrupted else 1
    cc, node_map, _ = conceptnet_subgraph_to_cc(sub, embedding_dim, node_list)
    node_list_final = list(node_map.keys())
    node_texts = _build_node_texts(sub, node_list)

    if corrupted and u_node in node_map and replacement in node_map:
        query_idx = node_map[u_node]
        target_idx = node_map[replacement]
    else:
        valid_indices = list(node_map.values())
        query_idx = valid_indices[0]
        target_idx = valid_indices[min(1, len(valid_indices) - 1)]

    meta = {
        'task_type': 'kg_consistency',
        'node_texts': node_texts,
        'corrupted': corrupted,
        'num_nodes': len(node_list_final),
        'num_classes': 2,
    }
    return cc, query_idx, target_idx, answer, meta


# ---------------------------------------------------------------------------
# v10 Task 8: Redesigned structural analogy
# ---------------------------------------------------------------------------


def generate_kg_analogy_task_v10(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 10,
    max_nodes: int = 30,
) -> tuple | None:
    """Redesigned analogy task using relation-type structural matching.

    Extracts two subgraphs. Computes structural similarity based on
    relation-type distribution overlap AND degree distribution similarity.
    Classes: 0=not_analogous, 1=partial, 2=analogous.
    """
    from src.cell_complex.cell_complex import CellComplex as CC
    from src.benchmarks.topological_tasks import auto_fill_triangles
    from collections import Counter

    sub_a, nlist_a = _safe_extract(G, min_nodes, max_nodes)
    sub_b, nlist_b = _safe_extract(G, min_nodes, max_nodes)

    # Compute relation-type distribution overlap (Jaccard)
    rels_a = Counter(
        data.get("relation", "RelatedTo") for _, _, data in sub_a.edges(data=True)
    )
    rels_b = Counter(
        data.get("relation", "RelatedTo") for _, _, data in sub_b.edges(data=True)
    )
    all_rel_keys = set(rels_a.keys()) | set(rels_b.keys())
    if all_rel_keys:
        intersection = sum(min(rels_a.get(k, 0), rels_b.get(k, 0)) for k in all_rel_keys)
        union = sum(max(rels_a.get(k, 0), rels_b.get(k, 0)) for k in all_rel_keys)
        rel_overlap = intersection / max(union, 1)
    else:
        rel_overlap = 0.0

    # Compute degree distribution similarity
    degs_a = sorted(dict(sub_a.degree()).values())
    degs_b = sorted(dict(sub_b.degree()).values())
    if degs_a and degs_b:
        max_deg = max(max(degs_a), max(degs_b), 1)
        norm_a = [d / max_deg for d in degs_a]
        norm_b = [d / max_deg for d in degs_b]
        mean_a = sum(norm_a) / len(norm_a)
        mean_b = sum(norm_b) / len(norm_b)
        deg_sim = 1.0 - abs(mean_a - mean_b)
    else:
        deg_sim = 0.0

    # Combined score
    score = 0.6 * rel_overlap + 0.4 * deg_sim

    if score > 0.6:
        answer = 2  # analogous
    elif score > 0.35:
        answer = 1  # partial
    else:
        answer = 0  # not analogous

    # Build merged graph
    merged = nx.Graph()
    for node in sub_a.nodes():
        merged.add_node(f"a_{node}", prefix="a", original=node)
    for u, v, data in sub_a.edges(data=True):
        merged.add_edge(f"a_{u}", f"a_{v}", **data)
    for node in sub_b.nodes():
        merged.add_node(f"b_{node}", prefix="b", original=node)
    for u, v, data in sub_b.edges(data=True):
        merged.add_edge(f"b_{u}", f"b_{v}", **data)

    # Build CellComplex
    cc = CC(embedding_dim=embedding_dim)
    node_map = {}
    merged_nodes = list(merged.nodes())
    merged_degrees = dict(merged.degree())
    merged_max_deg = max(merged_degrees.values()) if merged_degrees else 1
    merged_max_deg = max(merged_max_deg, 1)
    merged_clust = nx.clustering(merged)

    for mnode in merged_nodes:
        emb = torch.randn(embedding_dim) * 0.01
        emb[0] = merged_degrees[mnode] / merged_max_deg
        emb[1] = merged_clust[mnode]
        cc_idx = cc.add_0_cell(emb, "node")
        node_map[mnode] = cc_idx

    cc.node_texts = []
    for mnode in merged_nodes:
        original = merged.nodes[mnode].get("original", mnode)
        prefix = merged.nodes[mnode].get("prefix", "")
        cc.node_texts.append(f"{prefix}: {concept_to_text(original)}")

    from src.data.conceptnet import RELATION_CATEGORIES
    rel_to_idx = {r: i for i, r in enumerate(RELATION_CATEGORIES)}
    n_rels = len(RELATION_CATEGORIES)

    for u, v, data in merged.edges(data=True):
        emb = torch.randn(embedding_dim) * 0.01
        weight = data.get("weight", 1.0)
        relation = data.get("relation", "RelatedTo")
        emb[0] = weight / 10.0
        rel_idx = rel_to_idx.get(relation, rel_to_idx.get("RelatedTo", 0))
        if 1 + rel_idx < embedding_dim:
            emb[1:min(1 + n_rels, embedding_dim)] = 0.0
            emb[1 + rel_idx] = 1.0
        cc.add_1_cell(node_map[u], node_map[v], emb, relation)

    auto_fill_triangles(cc)

    query_node = f"a_{nlist_a[0]}"
    target_node = f"b_{nlist_b[0]}"
    query_idx = node_map[query_node]
    target_idx = node_map[target_node]

    meta = {
        'task_type': 'kg_analogy',
        'subgraph_a_size': sub_a.number_of_nodes(),
        'subgraph_b_size': sub_b.number_of_nodes(),
        'rel_overlap': rel_overlap,
        'deg_similarity': deg_sim,
        'combined_score': score,
        'num_classes': 3,
    }
    return cc, query_idx, target_idx, answer, meta


# ---------------------------------------------------------------------------
# v10 Task 9: Causal chain coherence
# ---------------------------------------------------------------------------

CAUSAL_CHAIN_CLASSES: list[str] = ["broken", "coherent", "incoherent"]
assert len(CAUSAL_CHAIN_CLASSES) == 3

_CAUSAL_RELATIONS = {"Causes", "UsedFor", "CapableOf", "HasPrerequisite",
                     "MotivatedByGoal", "CausesDesire"}


def generate_kg_causal_chain_task(
    G: nx.Graph,
    embedding_dim: int,
    min_nodes: int = 20,
    max_nodes: int = 50,
) -> tuple | None:
    """Generate causal chain coherence task with balanced class sampling.

    Picks a target class first (uniform over CAUSAL_CHAIN_CLASSES), then
    searches for a matching chain. This prevents "incoherent" from
    dominating the dataset.
    """
    target_class = random.choice(CAUSAL_CHAIN_CLASSES)

    for _ in range(_MAX_RETRIES):
        sub, node_list = _safe_extract(G, min_nodes, max_nodes)
        nodes = list(sub.nodes())
        random.shuffle(nodes)

        # Collect all 2-hop chains and classify them
        chains: list[tuple[str, str, str, str, str, str]] = []
        for a in nodes:
            neighbors_a = list(sub.successors(a) if sub.is_directed() else sub.neighbors(a))
            for b in neighbors_a:
                rel_ab = sub[a][b].get('relation', '')
                successors_b = list(sub.successors(b) if sub.is_directed() else sub.neighbors(b))
                for c in successors_b:
                    if c == a:
                        continue
                    rel_bc = sub[b][c].get('relation', '')

                    ab_causal = rel_ab in _CAUSAL_RELATIONS
                    bc_causal = rel_bc in _CAUSAL_RELATIONS

                    if ab_causal and bc_causal and rel_ab == rel_bc:
                        class_name = "coherent"
                    elif ab_causal and bc_causal and rel_ab != rel_bc:
                        class_name = "incoherent"
                    elif ab_causal != bc_causal:
                        class_name = "broken"
                    else:
                        class_name = "incoherent"

                    chains.append((a, b, c, rel_ab, rel_bc, class_name))

        if not chains:
            continue

        matching = [ch for ch in chains if ch[5] == target_class]
        if not matching:
            matching = chains

        a, b, c, rel_ab, rel_bc, class_name = random.choice(matching)
        answer = CAUSAL_CHAIN_CLASSES.index(class_name)

        cc, node_map, _ = conceptnet_subgraph_to_cc(sub, embedding_dim, node_list)
        query_idx = node_map[a]
        target_idx = node_map[c]

        node_texts = _build_node_texts(sub, node_list)

        meta = {
            'task_type': 'kg_causal_chain',
            'node_texts': node_texts,
            'chain': [a, b, c],
            'relations': [rel_ab, rel_bc],
            'class_name': class_name,
            'num_classes': len(CAUSAL_CHAIN_CLASSES),
        }
        return cc, query_idx, target_idx, answer, meta

    return None
