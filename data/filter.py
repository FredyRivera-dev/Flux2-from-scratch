### As a security measure in data preparation, we will 
### discard images that contain children or similar terms.

import re

ambiguous_terms = {
    "minor": [
        "minor key", "minor chord", "minor scale", "minor league",
        "minor detail", "minor issue", "minor problem", "minor injury",
        "minor character", "minor role", "asia minor", "ursa minor"
    ],

    "pupil": [
        "eye pupil", "pupil of the eye", "dilated pupil", "pupil dilation",
        "pupil size", "pupil response"
    ],

    "kid": [
        "kid gloves", "kid leather", "kid skin", "handle with kid gloves"
    ],

    "young": [
        "young adult", "young professional", "young man", "young woman",
        "young entrepreneur", "young artist", "young at heart",
        "die young", "too young", "young love", "young couple"
    ],

    "boy": [
        "boy band", "oh boy", "boy toy", "boy scout" , "altar boy",
        "cow boy", "play boy", "pool boy", "delivery boy", "office boy",
        "bad boy", "golden boy", "poster boy", "whipping boy",
        "boy next door", "mama's boy", "boy wonder", "boy genius"
    ],
    
    "girl": [
        "girl power", "girl boss", "party girl", "it girl", "cover girl",
        "girl next door", "working girl", "material girl", "bond girl",
        "pin up girl", "call girl", "show girl", "girl gang",
        "girl group", "girl friday", "golden girl"
    ],

    "student": [
        "university student", "college student", "graduate student",
        "medical student", "law student", "phd student", "doctoral student",
        "adult student", "mature student", "postgraduate student"
    ],
    
    "freshman": [
        "college freshman", "university freshman", "freshman year"
    ],

    "junior": [
        "junior developer", "junior engineer", "junior analyst",
        "junior partner", "junior level", "junior position",
        "junior size", "junior year"
    ],

    "lad": [
        "young lad", "lad culture", "lad mag", "top lad"
    ],
    
    "lass": [
        "young lass", "bonnie lass"
    ],

    "youth": [
        "youth culture", "youth movement", "fountain of youth",
        "youth hostel", "youth organization"
    ],
}

adult_indicators = [
    "professional", "career", "job", "work", "employee", "worker",
    "entrepreneur", "ceo", "manager", "director", "executive",

    "university", "college", "graduate", "phd", "doctorate", "postgraduate",
    "bachelor", "master's degree", "mba",

    "adult", "grown", "mature", "21+", "18+", "over 18", "over 21",
    "of age", "legal age",

    "husband", "wife", "married", "spouse", "partner", "dating",
    "engaged", "engagement", "wedding", "bride", "groom",

    "model", "actor", "actress", "performer", "artist",

    "vote", "voting", "voter", "drink", "drinking", "bar", "club",
]

strong_child_indicators = [
    r"\b([0-9]|1[0-7])\s*year[s]?\s*old\b",
    r"\b([0-9]|1[0-7])\s*yr[s]?\s*old\b",
    r"\bage\s*([0-9]|1[0-7])\b",

    "elementary", "primary school", "middle school", "high school",
    "kindergarten", "preschool", "daycare", "nursery",

    "playground", "recess", "school bus", "backpack",
    "homework", "report card", "parent teacher",

    "my son", "my daughter", "our child", "her child", "his child",
]

child_related_terms = [
    "child", "children", "kid", "kids", "kiddo", "kiddos",
    "minor", "minors", "youth", "youths",

    "baby", "babies", "infant", "infants", "newborn", "newborns",
    "toddler", "toddlers", "tot", "tots",

    "preschooler", "preschoolers", "kindergartener", "kindergarteners",
    "little boy", "little girl", "little one", "little ones",

    "schoolchild", "schoolchildren", "schoolboy", "schoolgirl",
    "elementary student", "grade schooler", "pupil", "pupils",

    "preteen", "preteens", "pre-teen", "pre-teens",
    "tween", "tweens",

    "teen", "teens", "teenager", "teenagers", "adolescent", "adolescents",
    "teenage boy", "teenage girl", "teen boy", "teen girl",

    "son", "daughter", "grandson", "granddaughter",
    "nephew", "niece", "godson", "goddaughter",
    "stepson", "stepdaughter",

    "youngster", "youngsters", "young one", "young ones",
    "young boy", "young girl", "young person",
    "lad", "lass", "boy", "girl",

    "student", "students", "pupil", "pupils",
    "schoolkid", "schoolkids",

    "kindergarten", "first grade", "second grade", "third grade",
    "fourth grade", "fifth grade", "sixth grade",
    "seventh grade", "eighth grade", "freshman", "sophomore",

    "little", "tiny", "small child", "wee one",

    "under 18", "underage", "juvenile", "juveniles",
    "under age", "underaged",

    "childs", "childrens", "babys",

    "child's", "children's", "kids'", "baby's", "babies'",
]

def has_adult_context(text: str) -> bool:
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in adult_indicators)

def has_strong_child_indicator(text: str) -> bool:

    text_lower = text.lower()

    for pattern in strong_child_indicators[:3]:
        if re.search(pattern, text_lower):
            return True

    for indicator in strong_child_indicators[3:]:
        if indicator in text_lower:
            return True
    
    return False

def is_false_positive(text: str, term: str) -> bool:
    text_lower = text.lower()

    if has_strong_child_indicator(text):
        return False

    if term in ambiguous_terms:
        for context in ambiguous_terms[term]:
            if context in text_lower:
                return True

    if has_adult_context(text):
        if term in ["young", "student", "freshman", "junior", "boy", "girl", "lad", "lass"]:
            return True

    return False

def contains_child_reference(text: str, terms_list: list) -> bool:    
    text_lower = text.lower()

    if has_strong_child_indicator(text):
        return True

    for term in terms_list:
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        if re.search(pattern, text_lower):
            if not is_false_positive(text, term.lower()):
                return True
    
    return False

def test_filter():
    test_cases = [
        # Cases that MUST be filtered (minors)
        ("a child playing in the park", True),
        ("10 year old boy with a bicycle", True),
        ("elementary school classroom", True),
        ("teenager doing homework", True),
        
        # Cases that should NOT be leaked (false positives)
        ("jazz music in a minor key", False),
        ("young professional woman at work", False),
        ("college student studying for exams", False),
        ("boy band performing on stage", False),
        ("girl power feminist movement", False),
        ("dilated pupil in the eye", False),
        ("junior developer coding", False),
        ("university freshman orientation", False),
        
        # Ambiguous cases with an adult context
        ("young couple getting married", False),
        ("adult student learning guitar", False),
    ]
    
    print("Testing filter:")
    for text, should_filter in test_cases:
        result = contains_child_reference(text, child_related_terms)
        status = "✓" if result == should_filter else "X"
        print(f"{status} '{text}' -> Filtered: {result} (Expected: {should_filter})")

if __name__ == "__main__":
    test_filter()