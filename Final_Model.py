from collections import defaultdict
from itertools import combinations
import spacy
import re
from PIL import Image
from pytesseract import pytesseract
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import os

import pandas as pd

# ------------------------------
# VIP NER Setup
# ------------------------------
nlp = spacy.load("./vip_ner_model")
vip_names = ["Sundar Pichai", "Virat Kohli", "Shah Rukh Khan", "Priyanka Chopra Jonas", "A. R. Rahman"]

# ------------------------------
# OCR Setup
# ------------------------------
path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.tesseract_cmd = path_to_tesseract

# ------------------------------
# Text cleaning and VIP detection
# ------------------------------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"\@\w+|\#", '', text)
    text = re.sub(r"[^A-Za-z0-9\s]", '', text)
    return text.lower().strip()

def extract_hashtags(text):
    return re.findall(r"#(\w+)", text)

def detect_vip_entities_and_hashtags(post_id, text):
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    hashtags = extract_hashtags(text)
    vip_hashtags = []
    for vip in vip_names:
        vip_clean = vip.lower().replace(" ", "")
        for tag in hashtags:
            if vip_clean in tag.lower():
                vip_hashtags.append("#" + tag)
    is_vip_related = any(ent['label'] == "PERSON" and ent['text'] in vip_names for ent in entities) or len(vip_hashtags) > 0
    return {
        "post_id": post_id,
        "text": text,
        "entities": entities,
        "vip_hashtags": vip_hashtags,
        "is_vip_related": is_vip_related
    }

def run_ocr(image_path):
    if not os.path.exists(image_path):
        return ""  # Return empty if file not found
    img = Image.open(image_path)
    return pytesseract.image_to_string(img).strip()

# ------------------------------
# Controversy & ML trust score
# ------------------------------
analyzer = SentimentIntensityAnalyzer()
controversy_keywords = ["exposed","scandal","leak","boycott","fraud","cheating","ban","corruption","hack","terror","racist","crime"]

def classify_controversy(text):
    text_clean = clean_text(text)
    sentiment = analyzer.polarity_scores(text_clean)
    keyword_hit = any(word in text_clean for word in controversy_keywords)
    strong_sentiment = abs(sentiment['compound']) > 0.5
    return "Controversial" if keyword_hit or strong_sentiment else "Non-Controversial"

# Simulated ML model
training_posts = [
    ("Celebrity X exposed in shocking scam!!!", 0.2),
    ("I love the new movie by Celebrity Y", 0.5),
    ("Massive boycott against Company Z reported by NDTV", 0.9),
    ("Government leak reveals secret deal", 0.8),
    ("Shocking truth about Celebrity Z", 0.1),
    ("Happy birthday to my friend", 0.5)
]
X_train_texts = [clean_text(text) for text, score in training_posts]
y_train = [score for text, score in training_posts]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_texts)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def predict_trust_score(text):
    X = vectorizer.transform([clean_text(text)])
    return round(model.predict(X)[0], 2)

def classify_real_fake_ml(post):
    score = predict_trust_score(post["combined_text"])
    label = "Real Controversy" if score >= 0.5 else "Fake Controversy"
    return label, score

# ------------------------------
# Collaborative Findings Functions
# ------------------------------
def text_similarity(text1, text2):
    return 1.0 if text1.lower() == text2.lower() else 0.0

def time_based_groups(posts, window_seconds=600):
    posts_sorted = sorted(posts, key=lambda x: x["timestamp"])
    clusters = []
    current_cluster = [posts_sorted[0]]
    for i in range(1, len(posts_sorted)):
        if posts_sorted[i]["timestamp"] - posts_sorted[i-1]["timestamp"] <= window_seconds:
            current_cluster.append(posts_sorted[i])
        else:
            if len(current_cluster) > 1:
                clusters.append(current_cluster)
            current_cluster = [posts_sorted[i]]
    if len(current_cluster) > 1:
        clusters.append(current_cluster)
    return clusters

def hashtag_groups(posts):
    hashtag_to_users = defaultdict(list)
    for p in posts:
        hashtags = tuple(sorted(p["hashtags"]))
        hashtag_to_users[hashtags].append(p["user_id"])
    groups = []
    for tags, users in hashtag_to_users.items():
        if len(users) > 1:
            groups.append({"hashtags": tags, "users": users})
    return groups

def interaction_graph(posts):
    edges = defaultdict(set)
    for p in posts:
        interactors = set(p.get("likes", []) + p.get("comments", []))
        for u1,u2 in combinations(interactors,2):
            edges[u1].add(u2)
            edges[u2].add(u1)
    return edges

def calculate_suspiciousness(group_posts, users):
    group_size = len(set(p["user_id"] for p in group_posts))
    text_similarity_score = sum(text_similarity(p1["combined_text"], p2["combined_text"]) for p1,p2 in combinations(group_posts,2))
    hashtag_score = sum(len(set(p1["hashtags"]) & set(p2["hashtags"])) for p1,p2 in combinations(group_posts,2))
    avg_account_age = sum(users[p["user_id"]]["account_age_days"] for p in group_posts) / group_size
    avg_followers = sum(users[p["user_id"]]["followers"] for p in group_posts) / group_size
    suspicious_score = (group_size*0.3 + text_similarity_score*0.3 + hashtag_score*0.2 +
                        (1/(avg_account_age+1))*0.1 + (1/(avg_followers+1))*0.1)
    return suspicious_score

# ------------------------------
# Full Integrated Pipeline
# ------------------------------
def process_posts(posts, users):
    results = []
    for post in posts:
        post_id = post["post_id"]
        text = post.get("caption","")
        media_path = post.get("media_path","")
        media_type = post.get("media_type","photo")

        vip_res = detect_vip_entities_and_hashtags(post_id, text)
        ocr_text = run_ocr(media_path) if media_type=="photo" and media_path else ""
        combined_text = text + " " + ocr_text if ocr_text else text

        controversy_label = classify_controversy(combined_text)
        classification, trust_score = classify_real_fake_ml({"post_id": post_id, "combined_text": combined_text})

        result = {
            "post_id": post_id,
            "is_vip_related": vip_res["is_vip_related"],
            "entities": vip_res.get("entities",[]),
            "vip_hashtags": vip_res.get("vip_hashtags",[]),
            "ocr_text": ocr_text,
            "combined_text": combined_text,
            "controversy_label": controversy_label,
            "trust_score": trust_score,
            "real_fake_classification": classification,
            "hashtags": extract_hashtags(text),
            "user_id": post.get("user_id"),
            "timestamp": post.get("timestamp",0)
        }
        results.append(result)

    # Collaborative Findings
    time_clusters = time_based_groups(results)
    hashtag_clusters = hashtag_groups(results)
    interaction_edges = interaction_graph(results)
    suspicious_scores = {tuple(p["post_id"] for p in cluster): calculate_suspiciousness(cluster, users) for cluster in time_clusters}

    return results, time_clusters, hashtag_clusters, interaction_edges, suspicious_scores

# ------------------------------
# Example Run with 6 posts
# ------------------------------
if __name__=="__main__":
    posts = [
        {"post_id":"post_1","user_id":"u1","caption":"Sundar Pichai announced new features for Google Chrome. #SundarPichai #Google","media_type":"photo","media_path":"ocr1.jpg","timestamp":1693747200,"likes":["u2","u3"],"comments":["u2","u4"]},
        {"post_id":"post_2","user_id":"u2","caption":"I love cooking pasta and trying new recipes every weekend. #Foodie","media_type":"photo","media_path":"ocr2.jpg","timestamp":1693747250,"likes":[],"comments":[]},
        {"post_id":"post_3","user_id":"u3","caption":"Massive boycott against Company Z reported on TV. #BreakingNews","media_type":"text","media_path":"","timestamp":1693747300,"likes":["u1"],"comments":["u2"]},
        {"post_id":"post_4","user_id":"u4","caption":"Virat Kohli scored a century in the finals! #ViratKohli #Cricket","media_type":"photo","media_path":"ocr3.jpg","timestamp":1693747350,"likes":["u1","u2"],"comments":["u3"]},
        {"post_id":"post_5","user_id":"u5","caption":"Just finished my painting today, art heals the soul. #ArtistLife","media_type":"text","media_path":"","timestamp":1693747400,"likes":["u2"],"comments":[]},
        {"post_id":"post_6","user_id":"u6","caption":"Shah Rukh Khan's new movie trailer leaked online! #ShahRukhKhan #Bollywood","media_type":"photo","media_path":"ocr4.jpg","timestamp":1693747450,"likes":["u3"],"comments":["u4"]}
    ]

    users = {
        "u1":{"account_age_days":10,"followers":5},
        "u2":{"account_age_days":20,"followers":10},
        "u3":{"account_age_days":300,"followers":500},
        "u4":{"account_age_days":15,"followers":8},
        "u5":{"account_age_days":100,"followers":50},
        "u6":{"account_age_days":5,"followers":2}
    }

    results, time_clusters, hashtag_clusters, interaction_edges, suspicious_scores = process_posts(posts, users)

    # Pretty Output
    for r in results:
        print(f"\nPost ID: {r['post_id']}")
        print(f"VIP-related: {r['is_vip_related']}")
        print(f"Entities: {r['entities']}")
        print(f"VIP Hashtags: {r['vip_hashtags']}")
        print(f"Controversy Label: {r['controversy_label']}")
        print(f"Real/Fake Classification: {r['real_fake_classification']}")
        print(f"Trust Score: {r['trust_score']}")
        print(f"Combined Text: {r['combined_text']}")

    print("\n📊 Collaborative Activity Clusters:")
    print("🕒 Time Clusters:", [[p["post_id"] for p in c] for c in time_clusters])
    print("#️⃣ Hashtag Clusters:", hashtag_clusters)
    print("🔗 Interaction Edges:", dict(interaction_edges))
    print("⚠️ Suspicious Scores:", suspicious_scores)



# Save report to text file
with open("final_report.txt", "w", encoding="utf-8") as f:
    for r in results:
        f.write(f"\nPost ID: {r['post_id']}\n")
        f.write(f"VIP-related: {r['is_vip_related']}\n")
        f.write(f"Entities: {r['entities']}\n")
        f.write(f"VIP Hashtags: {r['vip_hashtags']}\n")
        f.write(f"Controversy Label: {r['controversy_label']}\n")
        f.write(f"Real/Fake Classification: {r['real_fake_classification']}\n")
        f.write(f"Trust Score: {r['trust_score']}\n")
        f.write(f"Combined Text: {r['combined_text']}\n")

    f.write("\n📊 Collaborative Activity Clusters:\n")
    f.write(f"🕒 Time Clusters: {[[p['post_id'] for p in c] for c in time_clusters]}\n")
    f.write(f"#️⃣ Hashtag Clusters: {hashtag_clusters}\n")
    f.write(f"🔗 Interaction Edges: {dict(interaction_edges)}\n")
    f.write(f"⚠️ Suspicious Scores: {suspicious_scores}\n")

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("final_report.csv", index=False, encoding="utf-8")
