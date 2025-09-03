import spacy
import re

# ------------------------------
# Load trained VIP NER model
# ------------------------------
nlp = spacy.load("./vip_ner_model")

# Predefined VIP names
vip_names = ["Sundar Pichai", "Virat Kohli", "Shah Rukh Khan", "Priyanka Chopra Jonas", "A. R. Rahman"]

# ------------------------------
# Function to extract hashtags
# ------------------------------
def extract_hashtags(text):
    return re.findall(r"#(\w+)", text)

# ------------------------------
# Function to detect VIP entities and VIP hashtags
# ------------------------------
def detect_vip_entities_and_hashtags(post_id, text):
    doc = nlp(text)
    
    # Detect entities
    entities = []
    for ent in doc.ents:
        entities.append({"text": ent.text, "label": ent.label_})
    
    # Extract hashtags
    hashtags = extract_hashtags(text)
    
    # Match hashtags with VIP names
    vip_hashtags = []
    for vip in vip_names:
        vip_clean = vip.lower().replace(" ", "")
        for tag in hashtags:
            if vip_clean in tag.lower():
                vip_hashtags.append("#" + tag)
    
    # Check if post is VIP-related
    is_vip_related = any(ent['label'] == "PERSON" and ent['text'] in vip_names for ent in entities) or len(vip_hashtags) > 0
    
    return {
        "post_id": post_id,
        "text": text,
        "entities": entities,
        "vip_hashtags": vip_hashtags,
        "is_vip_related": is_vip_related
    }

# ------------------------------
# Sample posts (VIP + Non-VIP)
# ------------------------------
posts = {
    "post_1": "Sundar Pichai announced new features for Google Chrome. #SundarPichai #Google",
    "post_2": "Virat Kohli scored a century in the IPL match for Royal Challengers Bangalore. #ViratKohli #IPL",
    "post_3": "Shah Rukh Khan's new film Pathaan is breaking records at the box office. #ShahRukhKhan #Pathaan",
    "post_4": "Priyanka Chopra Jonas starred in the TV series Quantico. #PriyankaChopra #Quantico",
    "post_5": "A. R. Rahman composed the soundtrack for Slumdog Millionaire. #ARRahman #SlumdogMillionaire",
    "post_6": "The weather in New York is really nice today.",
    "post_7": "I love cooking pasta and trying new recipes every weekend. #Foodie",
    "post_8": "NASA announces new mission to study Mars atmosphere. #NASA #Space",
    "post_9": "The football match last night was amazing! #Sports",
    "post_10": "Reading a great book on machine learning. #AI #ML"
}

# ------------------------------
# Test all posts
# ------------------------------
for pid, text in posts.items():
    result = detect_vip_entities_and_hashtags(pid, text)
    print(f"\nPost ID: {result['post_id']}")
    print(f"Text: {result['text']}")
    print("VIP-related:", result['is_vip_related'])
    print("Detected Entities:", [f"{e['text']} ({e['label']})" for e in result['entities']])
    print("VIP-related Hashtags:", result['vip_hashtags'])
