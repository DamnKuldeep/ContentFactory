"""
Content Factory — Configuration, model registry, variety axes, sampling presets.

Everything extracted verbatim from the notebook. Zero modifications to any value.
"""

from __future__ import annotations


# ── Model registry ────────────────────────────────────────────────────────────

# Dense (non-MoE) models activate ALL weights per token, which performed best for this creative/visual work.
# Slugs + prices VERIFIED on openrouter.ai/models on 2026-06-17 (list price; real $ cost is also read live per call).
MODEL_OPTIONS = [
    "meta-llama/llama-3.3-70b-instruct",   # dense 70B   $0.10/$0.32   — prose generator + plan judge (best writer)
    "qwen/qwen3-vl-32b-instruct",          # dense VL 32B $0.104/$0.416 — scene planning + image prompts (only VL)
    "google/gemma-4-31b-it",               # dense 31B   $0.12/$0.35   — structured work, critics, judges
]

PRICES = {
    "meta-llama/llama-3.3-70b-instruct": (0.10, 0.32),
    "qwen/qwen3-vl-32b-instruct":        (0.104, 0.416),
    "google/gemma-4-31b-it":             (0.12, 0.35),
}

# Per-role defaults, chosen by what each model is best at. Generator family = Llama-3.3; the critic jury and the
# judges that grade a model's output are a DIFFERENT family (anti self-preference).
ROLE_MODELS = {
    # prose (story + script) — best writer
    "concept_director":  "meta-llama/llama-3.3-70b-instruct",
    "ideator":           "meta-llama/llama-3.3-70b-instruct",
    "story_drafter":     "meta-llama/llama-3.3-70b-instruct",
    "story_reviser":     "meta-llama/llama-3.3-70b-instruct",
    "story_showrunner":  "meta-llama/llama-3.3-70b-instruct",
    "interview_amateur": "meta-llama/llama-3.3-70b-instruct",
    "script_drafter":    "meta-llama/llama-3.3-70b-instruct",
    "script_reviser":    "meta-llama/llama-3.3-70b-instruct",
    "script_showrunner": "meta-llama/llama-3.3-70b-instruct",
    "plan_judge":        "meta-llama/llama-3.3-70b-instruct",   # judges the Qwen planner (different family)
    # vision-language (images)
    "scene_planner":     "qwen/qwen3-vl-32b-instruct",
    "prompt_writer":     "qwen/qwen3-vl-32b-instruct",
    "prompt_rewriter":   "qwen/qwen3-vl-32b-instruct",
    # structured work + reasoning judges — Gemma-4 is strongest here of the four
    "interview_expert":  "google/gemma-4-31b-it",
    "premise_judge":     "google/gemma-4-31b-it",
    "prompt_reviewer":   "google/gemma-4-31b-it",               # judges the Qwen writer (different family)
    "extractor":         "google/gemma-4-31b-it",
    "segmenter":         "meta-llama/llama-3.3-70b-instruct",  # Llama-3.3 places the cuts; we re-slice from its boundaries so narration stays verbatim by construction
    "style_selector":    "google/gemma-4-31b-it",
    "feedback_editor":   "google/gemma-4-31b-it",
    "plan_updater":      "google/gemma-4-31b-it",
    # critic jury — reliable TEXT critics (gemma-4 + llama-3.1), never the Llama-3.3 generator. critic_a/b judge
    # the STORY; critic_a/b/c judge the SCRIPT. The qwen-vl vision model is kept OFF the critic jury (it kept
    # emitting unusable verdicts on text); it does scene planning + image prompts instead.
    "critic_a":          "google/gemma-4-31b-it",
    "critic_b":          "meta-llama/llama-3.3-70b-instruct",
    "critic_c":          "google/gemma-4-31b-it",
}


def model_for(role: str) -> str:
    return ROLE_MODELS[role]


# Sampling presets (research: creative = hot + min_p; judge/mechanical = cold)
SAMPLING = {
    "ideate":     dict(temperature=1.15, min_p=0.05, presence_penalty=0.4),
    "create":     dict(temperature=1.00, min_p=0.06),
    "revise":     dict(temperature=0.70, min_p=0.05),
    "finalize":   dict(temperature=0.60, min_p=0.04),
    "judge":      dict(temperature=0.30),
    "mechanical": dict(temperature=0.30),
}

# Fixed, sensible loop counts (NOT user knobs)
LOOPS = {
    "min_interview":      1,   # amateur↔expert: ONE rich exchange is enough material; the amateur may leave after it
    "max_interview":      3,   # ...ceiling — it deepens only if round 1 wasn't enough, and exits early once it can imagine a great story
    "brief_rounds":       2,   # script: amateur↔director brief discussion
    "story_refine":       4,   # convergence loops (critics → reconcile → surgical revise); passed lanes retire
    "script_refine":      4,
    "plan_judge":         3,
    "prompt_review":      4,
    "segment_attempts":   3,   # we re-slice from the model's cuts (verbatim by construction); the model only needs to land the COUNT, usually in 1-2 tries, with a deterministic split/merge backstop
}
ACCEPT = {"story": 4.2, "script": 4.2, "plan": 4.2}   # (plan still uses a numeric bar; story/script now use the binary `satisfied` gate)


# ── Creative configuration ────────────────────────────────────────────────────

CONFIG = {
    "genre": "Fabricated dark micro-stories in the CREEPY-TRUE-STORY / unsolved-mystery / dark-history vein. They feel like they COULD be real. An uncanny, eerie, even seemingly-supernatural SURFACE is welcome (it makes them gripping) — but the ENGINE is human and plausible, OR left as a genuine open mystery. No worked magic systems, no incoherence.",
    "platform": "Instagram Reels / YouTube Shorts (vertical, 9:16)",
    "audience": "16–35",
    "story_len": {"low": 320, "target": 430, "high": 600},   # the rich SOURCE story (can run long)
    "script_len": {"low": 220, "target": 260, "high": 300},  # the spoken NARRATION (flowing) — stays within 220-300 words
    "scenes": {"low": 28, "high": 40, "words_per_scene": 8},  # 28-40 visual beats; aim ≈ one beat per ~8 spoken words (a natural middle of the band, never the floor or ceiling)

    "key_instructions": [
        "FEELS REAL OR REALLY UNEXPLAINED: a disturbing case, an eerie unsolved mystery, or a dark historical episode. An uncanny/eerie SURFACE is encouraged (a creepy hook, a strange coincidence, an unexplained detail). But the story's ENGINE is HUMAN and plausible — crime, exploitation, fraud, captivity, cruelty, a cover-up — OR deliberately left as a genuine open mystery. It may START seeming supernatural and turn out dark-but-human, or stay an ambiguous chill.",
        "NO worked magic and NO incoherence: never a fully-explained supernatural mechanism the plot runs on (e.g. 'a clock that literally freezes time for prisoners'), and never cause-and-effect that breaks the 'this could be real' spell. If something uncanny happens, it resolves into a human explanation or stays unexplained — it is never a working magic system.",
        "SPECIFIC, documentary detail: a year, a place, a name, a number, a price — so it feels like a real case.",
        "ONE clear spine that ESCALATES: one person, one pressure, a line crossed, a causal chain, a shocking turn or reveal.",
        "DRIVEN BY ACTIONS AND CONSEQUENCES: show the story through concrete EVENTS — choices made, things done, and what each one causes — not through stated feelings or mood. Emotion lands because of what HAPPENS, never because we are told someone felt dread.",
        "VARY THE SHAPE: do not let THIS story fall into the default kind (a string of unexplained deaths/disappearances tied to a recurring object). Its darkness can come from obsession, deception, an impostor, hubris, a hoax, a strange practice, greed, a discovery, or a rivalry — whatever the chosen world makes natural.",
        "PLAIN, FLOWING language — complete, connected sentences, the way a person tells a gripping true story aloud. NEVER choppy one-word fragments, never ornate or literary.",
        "Dark, taboo, or scandalous is welcome and good for variety, told SUGGESTIVELY — never graphic or explicit.",
        "END BY LANDING THE PAYOFF: finish on the consequence or final turn of the spine so it feels complete and resonant — the result of what happened. A single genuinely chilling open question is allowed, but NEVER a vague poetic fade and NEVER a call-to-action ('follow for more', 'like and subscribe') — those do not belong in the narration.",
        "VARY THE OPENER: the hook need not always be 'Did you know there was a … who …?'. Use whatever first line grabs hardest — a blunt shocking fact, a stark image, a short question, an impossible-sounding claim — and keep the payoff up front.",
        "Do NOT defame a real, named, living person; keep it period-historical and clearly fictional.",
    ],

    # GROUNDED story shapes — match the SHAPE + plausibility, never the content. Rotated so no single template is copied.
    "gold_examples": [
        ("In 1920s Pennsylvania, a dairy farmer named Adam woke one morning to find every cow on his farm had stopped "
         "giving milk. The farm was all he had, and it was dying. His wife had just given birth. Desperate, he milked "
         "her, bottled it, and sold it at market — and people went strangely wild for it. Demand grew, so he brought "
         "in another new mother, then another, until more than fifty women were kept on his farm and made to keep "
         "producing. He grew rich and respectable. Then one woman escaped and ran to the police, and it all came apart "
         "in a day. He was arrested on the spot. Years later he walked free, and no one knows where he went."),
        ("In 1931, a struggling model named Vera was told by a back-street doctor that she could have the smallest "
         "waist in the city if she gave up two of her lower ribs. She paid him nearly everything she had and let him "
         "do it in a rented room above a pharmacy. For a season she was the most photographed face in town, laced into "
         "dresses no one else could wear. Then the pain started, her breathing turned shallow, and the same doctor told "
         "her the ribs had to go back or she would not walk for long. She never appeared in a photograph again, and the "
         "studio that made her famous quietly burned her portfolio."),
        ("Along Route 66 in the 1950s, a roadside diner became famous for a single impossible-tasting burger that drew "
         "cars for miles. Truckers swore by it; families drove out just to say they'd had one. The only complaint was "
         "that no one could say what the meat actually was. When a new health inspector finally went down to the cellar, "
         "she found what the owner had been grinding into every order — and why the recipe had never once changed in "
         "eleven years. The diner closed the next morning. The owner was never charged, because by the time the police "
         "arrived the cellar was empty and scrubbed clean."),
        ("In 1974, families in a small Oregon town kept finding their group photos spoiled the same way: in every shot, "
         "one child stood slightly apart, blurred, half-turned as if about to walk out of frame. People said the camera "
         "was cursed. When a state investigator finally examined the old negatives, the truth was worse and entirely "
         "human — the same quiet man had stood at the edge of every gathering for years, and the children he stood "
         "nearest were the ones who later went missing. The 'blur' was him stepping back out of the picture. He was "
         "never identified, and the town still won't look too closely at its old photographs."),
    ],
    # The SPOKEN NARRATION style: hook-first, flowing, plain, ends on a hook. Match the VOICE, never the content.
    "gold_script": (
        "Did you know there was a man who stopped milking cows and started milking women instead? This happened in 1920 "
        "in Pennsylvania. A dairy farmer named Adam woke up one morning and every cow on his farm had stopped giving milk "
        "— not a single drop. He panicked, because the farm was the only money he had. His wife had just given birth, so "
        "in desperation he took her out to the barn and did something no one could imagine: he started milking her. He "
        "filled a whole bucket, took it to the market, and sold it — and people went wild for it. So he brought in another "
        "new mother, then another, until more than fifty women were kept on that farm and forced to keep producing. He "
        "became one of the richest men in the area. But it didn't last. One of the women escaped and ran straight to the "
        "police, and the truth came out in a single day. He was arrested on the spot, and every woman was freed. But he "
        "was released not long ago — and now no one knows where he is."),
    "gold_why": ("WHY THESE WORK: they feel like a REAL case or a real unsolved mystery — an eerie surface is fine (a "
                 "cursed-camera hook) as long as the engine is human (a predator) or genuinely unexplained; one spine "
                 "that escalates; plain flowing sentences a person would actually say aloud; specific detail (year, "
                 "place, name, number); an ending that lands or opens a loop. THE FAILURE TO AVOID: a worked magic "
                 "system or impossible mechanism the plot depends on (a clock that literally freezes time), incoherent "
                 "cause-and-effect, 'clever' machinery (detective puzzles, conspiracies), and choppy fragments. They also DRIVE ON EVENTS AND CONSEQUENCES (things happen and cause the next), not on stated feelings; MORE failures to avoid: repeating the same shape every time (unexplained deaths/disappearances pinned on a recurring object), endings that fizzle into a vague poetic image or a tacked-on 'follow for more', and stories that are all atmosphere with no concrete events."),

    "ban_list": ["a worked/explained magic system or real supernatural powers the plot runs on (a clock that literally "
                 "freezes time, a curse that actually kills)", "incoherent cause-and-effect that breaks the 'this could "
                 "be real' spell", "tired clichés: haunted-asylum tours, Ouija boards, a serial killer caught by a "
                 "clever detective, Bermuda-triangle disappearances, Jack-the-Ripper retellings, vague 'mad scientist "
                 "went too far' experiments", "engagement-bait or calls-to-action inside the narration ('follow for more', 'hit follow', 'like and subscribe')", "the overused template of a string of unexplained deaths or disappearances 'explained' by a recurring token object", "stories that are all mood and stated feelings with no concrete events, actions or consequences", "building the story's central engine on real-world identity-based atrocity (racial, ethnic or religious murder, genocide, lynching) — keep the darkness in individual human crime, obsession, deception, hubris, greed, or the uncanny"],
}


def good_story_block() -> str:
    g = CONFIG
    ex = "\n\n".join("GOLD STORY (match the SHAPE + grounded realism, never the content):\n  " + e for e in g["gold_examples"])
    rules = "\n".join("- " + s for s in g["key_instructions"])
    return (f"{ex}\n\n{g['gold_why']}\n\nRULES:\n{rules}\n"
            f"STORY LENGTH: {g['story_len']['low']}–{g['story_len']['high']} words (aim ~{g['story_len']['target']}); "
            f"the story is the rich SOURCE — it can run long.")


def good_script_block() -> str:
    g = CONFIG
    return ("GOLD NARRATION — match this SPOKEN VOICE and ENERGY (flowing, plain, hook-first), NOT its exact opener or content:\n  "
            + g["gold_script"] +
            "\n\nHOOK: open with whatever first line grabs hardest and puts the payoff up front. VARY THE OPENER — the gold "
            "uses 'Did you know there was a … who …?', but that is only ONE option; also use a blunt shocking statement, a stark "
            "concrete image, a flat impossible-sounding claim, or a short question. Pick whichever opener fits THIS story instead of defaulting to one formula."
            "\nENDING: land the consequence/turn of the spine so it feels complete; one genuine chilling open question is fine, but "
            "never a vague poetic fade and never a call-to-action."
            f"\n\nSCRIPT LENGTH: {g['script_len']['low']}–{g['script_len']['high']} spoken words (aim ~{g['script_len']['target']}).")


def banned_block() -> str:
    return "AVOID entirely: " + "; ".join(CONFIG["ban_list"]) + "."


# ── Aspect ratio dimensions ───────────────────────────────────────────────────

DIMS = {"9:16": [752, 1328], "1:1": [1024, 1024], "16:9": [1328, 752], "4:5": [912, 1136]}


# ── Variety axes ──────────────────────────────────────────────────────────────

VARIETY_AXES = {
    # Eras are naturally finite -> keep this list the SMALLEST; spread evenly antiquity -> 1990s.
    # Expanded by 5 high-signal ancient/transitional eras for deeper historical range without bloating the list.
    "era": [
        "the 1400s", "the 1500s", "the 1600s", "the 1700s",
        "the early 1800s", "the mid 1800s", "the late 1800s",
        "the 1890s", "the 1900s", "the 1910s", "the 1920s", "the 1930s", "the 1940s",
        "the 1950s", "the 1960s", "the 1970s", "the 1980s", "the 1990s"
    ],

    # PLACE: large + niche + globally spread (every continent). Mostly GEOGRAPHIC/cultural so they fit many eras.
    "region": [
        # Europe (+3)
        "a Cornish tin-mining village", "a Hebridean island croft", "a Highland Scottish glen", "a Welsh slate-quarry valley",
        "a Sicilian hill town", "an Andalusian olive town", "a Venetian lagoon island", "a Flemish weaving town",
        "a Bavarian forest village", "a Harz mountain mining town", "a Norwegian fjord hamlet", "an Icelandic coastal settlement",
        "a Carpathian mountain village", "a Transylvanian market town", "a Russian taiga village",
        "a Basque coastal village", "a Provençal hill town", "a Bohemian forest hamlet",
        # Middle East & Central Asia (+1)
        "a Persian caravan city", "an Anatolian highland town", "a Yemeni mountain town", "a Silk Road oasis",
        "a Cairo bazaar quarter", "a Damascus old-city quarter", "a Bukhara madrasa town", "a Bedouin desert camp",
        "an Armenian highland monastery town",
        # Africa (+3)
        "a Saharan salt-caravan oasis", "a Swahili-coast trading port", "an Ethiopian highland monastery town",
        "a West African river kingdom", "a Malagasy highland village", "a Cape frontier farm", "a Nile delta town",
        "a trans-Saharan way-station", "a Barbary-coast port", "a Kalahari-edge settlement",
        "a Great Zimbabwe stone settlement", "an Ashanti gold-trading town", "a Dogon cliffside village",
        # Asia (+2)
        "a Himalayan border monastery town", "a Japanese mountain hot-spring village", "a Hokkaido fishing port",
        "a Korean mountain temple village", "a Qing-dynasty canal town", "a Tibetan plateau outpost", "a Bengali river-delta town",
        "a Mekong delta village", "a Javanese plantation town", "a Burmese teak-logging camp", "a Malabar-coast spice port",
        "a Mughal walled city", "a Mongolian steppe encampment", "a Ceylon tea estate", "a Sumatran rainforest outpost",
        "a Khmer temple village", "a Ryukyu island trading port",
        # Americas (+2)
        "an Appalachian coal-mining hollow", "a Mississippi river town", "a New England whaling port", "a Quebec fur-trade post",
        "a Klondike gold camp", "a Patagonian sheep station", "an Andean silver-mining town", "a Yucatan henequen estate",
        "an Amazon rubber-tapping station", "a Caribbean sugar plantation", "a Newfoundland fishing outport",
        "a Louisiana bayou settlement", "a Nevada silver boomtown", "a Brazilian colonial gold town", "a Chilean nitrate town",
        "a Haitian mountain village", "a Sonoran desert mission", "a prairie homestead",
        "a Pacific Northwest cedar longhouse village", "an Arctic Inuit hunting camp",
        # Oceania (+1)
        "a Tasmanian penal coast", "an outback Australian station", "a New Zealand gold-rush town", "a Pacific atoll mission",
        "a Torres Strait pearling station", "a New Guinea highland post",
        "a Maori fortified pa village"
    ],

    # DOMAIN: the field the darkness grows in. Large + wide so we are NOT biased to industrial/labour.
    "domain": [
        "medicine", "anatomy & dissection", "surgery & barber-surgeons", "apothecaries & poisons", "alchemy & early chemistry",
        "midwifery", "dentistry", "asylums & psychiatry", "plague & quarantine work", "the morgue & coroners",
        "the church", "monastic life", "the inquisition & heresy courts", "the seminary",
        "natural science", "astronomy & almanacs", "cartography & surveying", "academia & the university",
        "the library & archive", "translation & scribes",
        "art & portraiture", "the theatre & opera", "the circus & fairs", "the menagerie", "photography", "taxidermy",
        "music & instrument-making", "clock & watchmaking", "printing & books",
        "law & the courts", "policing & detection", "crime & punishment", "smuggling & contraband", "espionage & couriers",
        "diplomacy & the court", "banking & debt", "money-lending & usury", "insurance & shipping risk", "tax-farming",
        "commerce & trade", "the spice trade", "the fur trade", "shipping & the sea", "whaling & sealing", "deep-sea fishing",
        "pearl-diving", "lighthouse-keeping", "mining & ore", "gold & gem prospecting", "salt works", "glassblowing",
        "bell-founding", "tanning & leather", "brewing & distilling", "textiles & weaving", "forestry & logging",
        "the funerary trade", "embalming & undertaking",
        # New
        "herbalism & folk healing", "body snatching & resurrectionists", "art forgery & relic trading",
        "piracy & privateering", "counterfeiting & document forgery", "phrenology & pseudoscience",
        "spiritualism & séance fraud", "university secret societies", "theatre censorship & scandals",
        "journalism & scandal sheets", "temperance movement underbelly"
    ],

    # MILIEU: the specific social unit/place the story lives in. Large + wide; complements (not duplicates) DOMAIN.
    "milieu": [
        "a guild hall", "a convent", "a nunnery", "a monastery scriptorium", "a cathedral chapter", "a seminary",
        "a boys' boarding school", "a girls' finishing school", "an asylum for the insane", "a sanatorium", "a leper colony",
        "a quarantine station", "a workhouse", "an almshouse", "a foundling hospital", "a charity hospital ward",
        "an anatomy theatre", "a coroner's court", "a debtors' prison", "a prison hulk", "a penal colony",
        "a military garrison", "a naval frigate", "a whaling ship", "a merchant caravan", "a pilgrim hostel",
        "a remote lighthouse", "a customs house", "a coaching inn", "a country manor", "a tenement block",
        "a textile mill", "a coal pit", "a quarry camp", "a logging camp", "a fur-trade fort", "a mission station",
        "a printing house", "a clockmaker's workshop", "a portrait studio", "a photographer's studio",
        "a taxidermist's workshop", "an apothecary's shop", "a perfumer's atelier", "a travelling circus",
        "an opera company", "a theatre troupe", "a gentlemen's club", "a masonic lodge", "a seance circle",
        "a royal household", "an embassy", "a smugglers' cove", "a gambling den", "a riverboat", "a polar expedition",
        "an isolated village", "a plantation great-house",
        # New
        "a frontier saloon", "a company mining town", "a Victorian boarding house", "a public execution square",
        "a secret society meeting room", "a hospital dissecting room", "a ship chandlery", "a noble's hunting lodge",
        "a prison chapel", "a lighthouse keeper's quarters", "a market square at first light", "a dueling meadow at dawn"
    ],

    # FLAVOR: the KIND of dark. Wide so the engine is not always a death/disappearance.
    "flavor": [
        "cold-crime", "true-crime mystery", "tragic", "scandalous (suggestive, not explicit)", "a disturbing real-sounding case",
        "absurd-but-true", "grim irony", "exploitation & greed", "claustrophobic", "a quiet betrayal", "obsession curdling",
        "a slow-unravelling cover-up", "a hoax that grows out of control", "a reputation built on a lie",
        "a discovery that should have stayed buried", "a rivalry turning poisonous", "an addiction or compulsion",
        "a strange practice taken too far", "mass delusion or a craze", "a forbidden craft or trade", "an impostor among them",
        "hubris meeting its bill", "a loyalty that curdles", "a performance that goes wrong", "a swindle unravelling",
        "a forged identity", "a double life", "a healer who harms", "a protector who preys", "a benefactor with a hidden price",
        "a kindness used as control", "an inheritance turned deadly", "a feud across generations", "a secret child",
        "a scapegoat chosen", "a wrongful conviction", "a mob's certainty", "a shared delusion", "a miracle exposed as fraud",
        "a quarantine turned cruel",
        # New
        "gaslighting and slow manipulation", "institutional cruelty disguised as care", "a cult of personality that devours",
        "blackmail that tightens like a noose", "whistleblower systematically destroyed", "scientific ambition that corrupts",
        "artistic obsession that consumes its creator", "community complicity in quiet atrocity",
        "a secret society whose loyalty test destroys lives", "inheritance dispute that turns families into enemies",
        "false memory weaponized against the innocent", "loyalty that becomes a cage"
    ],

    # STRUCTURE: the plot shape.
    "structure": [
        "rise then fall", "one fatal decision", "a slow-creeping wrongness", "a mystery with an open loop",
        "ordinary -> monstrous escalation", "a debt that comes due", "a secret kept too long", "a bargain and its price",
        "a pattern no one will admit", "a small lie that grows teeth", "a discovery and its price", "a hoax that consumes its maker",
        "a contest escalating to ruin", "a talent that becomes a trap", "a rescue that makes things worse",
        "a cure worse than the affliction", "an obsession that swallows a life", "a lie that needs more lies",
        "a favour that becomes a leash", "a secret that outlives its keeper", "a confession that comes too late",
        "a witness slowly silenced", "a cover-up that spreads", "a hunt that consumes the hunter",
        "a protector who becomes the threat", "a benefactor revealed as predator", "a healer who becomes the disease",
        "an impostor who cannot stop", "a performance that cannot end", "a craze that burns out in tragedy",
        "a search that finds the wrong thing", "a return that should not have happened",
        # New
        "a friendship curdling into betrayal", "a gift that carries a terrible price", "the slow realization of one's own complicity",
        "a mistake that snowballs into irreversible ruin", "a hidden lineage revealed at the worst possible moment",
        "a trial rigged to destroy the innocent", "an alliance formed in desperation that backfires catastrophically",
        "a prophecy of doom that fulfills itself through the very fear it creates", "a scapegoat who turns the accusation back on the accusers"
    ],

    # REGISTER: the telling voice. Kept SMALL (few genuinely distinct narration tones).
    "register": [
        "matter-of-fact and chilling", "mournful", "uneasy", "darkly wry", "hushed and confiding", "clipped and clinical",
        "like a rumour passed along", "cold and documentary", "like a coroner's report", "like a court testimony",
        "like a letter never sent", "like a deathbed account", "like village gossip", "plain and unsettling",
        "measured and ominous", "intimate and disturbing", "dry and forensic", "like a case file read aloud",
        # New
        "like a half-remembered nightmare", "sharp-tongued and bitterly ironic", "like a faded newspaper editorial",
        "hauntingly detached, as if the teller is still in shock"
    ],

    # MOTIF: a recurring object or eerie detail -> the LARGEST list.
    "motif": [
        "a bell", "a bell that rings on its own", "a ledger of names", "a ledger with a crossed-out name", "a church register",
        "a baptismal record", "a missing page", "a will with an erasure", "a codicil", "a debt-book", "a pawn ticket",
        "a one-way ticket", "a hotel key for a room never let", "an iron key", "a key with no lock", "a lock with no key",
        "a locked door", "a bricked-up doorway", "a trapdoor", "a hidden room", "a buried tin box", "a mirror",
        "a mirror turned to the wall", "a portrait painted over", "a painted portrait", "a portrait with the eyes cut out",
        "a death mask", "a plague doctor's mask", "a mask", "a wax doll", "a doll with no face", "a marionette",
        "a ventriloquist's dummy", "a child's shoe", "a single glove", "a lock of hair", "a braid of hair in a locket",
        "a mourning brooch", "a black-edged letter", "an unsigned letter", "a sealed envelope", "a torn photograph",
        "a daguerreotype", "a tin of old photographs", "a wax-cylinder recording", "a telegram", "a coded diary", "a cipher",
        "a map with an X", "a ship's logbook", "a confession in the margin", "a recipe in an unknown hand",
        "a herbal with one plant marked poison", "a poison bottle", "a vial of laudanum", "an apothecary's scale",
        "a surgeon's kit", "a glass eye", "a set of false teeth", "a jar of specimens", "a preserved hand", "a row of jars",
        "a taxidermied bird", "a scrimshaw carving", "a whalebone corset", "a snuffbox", "a calling card", "a signet ring",
        "a wedding ring", "a stopped clock", "a clock stopped at one hour", "a pocket watch", "a music box",
        "a music box that plays one tune", "a single lamp", "a candle burned to the stub", "a votive candle", "a reliquary",
        "a length of rope", "a length of chain", "a branding iron", "a set of manacles", "a tally carved in wood",
        "a name scratched on a wall", "a dry well", "a foghorn", "a lighthouse lamp", "a ship in a bottle", "a quarantine flag",
        "a too-small grave", "a row of identical graves", "a gravestone with the wrong date", "an empty chair kept set",
        # New (intimate + institutional)
        "a blood-stained handkerchief", "a broken pocket watch", "a faded love letter", "a rusted locket with no picture",
        "a half-burned diary", "a pair of child's spectacles", "a military discharge paper with a hole punched through",
        "a false identity card", "a pressed poisonous flower in a bible", "a jar of extracted teeth",
        "a taxidermied cat with glass eyes", "a phonograph horn that still smells of smoke",
        "a typewriter with jammed keys and a half-finished page", "a smuggler's false-bottom cigar box",
        "a hangman's noose fragment", "a judge's cracked gavel", "a surgeon's blood-crusted apron",
        "a nun's rosary with three beads missing", "a gambler's loaded dice", "a spy's brass cipher wheel",
        "an empty grave plot with the marker already carved", "a quarantine bell that rings at odd hours",
        "a leper's wooden warning rattle", "a foundling's tiny silver token left on a doorstep",
        "a widow's black veil still smelling of lavender", "a pair of manacles with one lock forced open"
    ],
}
