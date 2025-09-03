import spacy
from spacy.tokens import DocBin
from spacy.util import minibatch, compounding
import random

# ------------------------------
# Step 1: Prepare training data
# ------------------------------
# Each item: (text, {"entities": [(start, end, label), ...]})
TRAIN_DATA = [
    # Sundar Pichai
    ("Sundar Pichai is the CEO of Google.", {"entities": [(0, 12, "PERSON"), (24, 28, "TITLE"), (32, 38, "ORG")]}),
    ("Pichai was born in Madurai, Tamil Nadu.", {"entities": [(0, 6, "PERSON"), (19, 32, "LOCATION")]}),
    ("He leads products like Google Chrome and Gmail.", {"entities": [(22, 35, "PRODUCT"), (40, 45, "PRODUCT")]}),

    # Virat Kohli
    ("Virat Kohli is captain of the Indian national cricket team.", {"entities": [(0, 11, "PERSON"), (25, 67, "ORGANIZATION"), (15, 22, "TITLE")]}),
    ("He won the ICC ODI Player of the Year award.", {"entities": [(12, 39, "AWARD")]}),

    # Shah Rukh Khan
    ("Shah Rukh Khan acted in Dilwale Dulhania Le Jayenge.", {"entities": [(0, 14, "PERSON"), (24, 54, "FILM")]}),

    # Priyanka Chopra Jonas
    ("Priyanka Chopra Jonas starred in Quantico.", {"entities": [(0, 22, "PERSON"), (34, 42, "FILM")]}),

    # A.R. Rahman
    ("A. R. Rahman composed the soundtrack for Slumdog Millionaire.", {"entities": [(0, 12, "PERSON"), (40, 58, "FILM_SOUNDTRACK")]}),
]

# ------------------------------
# Step 2: Create a blank English model
# ------------------------------
nlp = spacy.blank("en")

# Add NER pipeline if not exists
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Add new labels
labels = ["PERSON", "ORG", "TITLE", "LOCATION", "DATE", "FILM", "AWARD", "SPOUSE", "SPORTING_EVENT", "PRODUCT", "FILM_SOUNDTRACK"]
for label in labels:
    ner.add_label(label)

# ------------------------------
# Step 3: Convert training data to spaCy format
# ------------------------------
db = DocBin()
for text, annot in TRAIN_DATA:
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in annot["entities"]:
        span = doc.char_span(start, end, label=label)
        if span is not None:
            ents.append(span)
    doc.ents = ents
    db.add(doc)

db.to_disk("./train.spacy")

# ------------------------------
# Step 4: Training Loop
# ------------------------------
# Load training data
from spacy.training.example import Example
import pathlib

# Training setup
n_iter = 30
optimizer = nlp.begin_training()

# Load examples
doc_bin = DocBin().from_disk("./train.spacy")
examples = [Example.from_dict(nlp.make_doc(doc.text), {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}) for doc in doc_bin.get_docs(nlp.vocab)]

# Training loop
for i in range(n_iter):
    random.shuffle(examples)
    losses = {}
    batches = minibatch(examples, size=compounding(2.0, 16.0, 1.5))
    for batch in batches:
        nlp.update(batch, sgd=optimizer, drop=0.2, losses=losses)
    print(f"Iteration {i+1}, Losses: {losses}")

# ------------------------------
# Step 5: Save the trained model
# ------------------------------
output_dir = pathlib.Path("./vip_ner_model")
nlp.to_disk(output_dir)
print("Trained NER model saved at:", output_dir)

# ------------------------------
# Step 6: Test the model
# ------------------------------
test_text = "Sundar Pichai announced new features for Google Chrome."
nlp2 = spacy.load("./vip_ner_model")
doc = nlp2(test_text)
for ent in doc.ents:
    print(ent.text, ent.label_)
