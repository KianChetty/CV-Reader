import spacy
import pickle
import random
import sys, fitz

#print("Hello World")

train_data = pickle.load(open('train_data (1).pkl', 'rb'))

nlp = spacy.blank('en')


def train_model(train_data):
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

    for _, annotation in train_data:
        for ent in annotation['entities']:
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

    with nlp.disable_pipes(*other_pipes):

        optimizer = nlp.begin_training()
        for itn in range(10):
            print("starting iteration " + str(itn))
            random.shuffle(train_data)
            losses = {}
            index = 0
            for text, annotations in train_data:
                try:
                    nlp.update([text], [annotations], drop=0.2, sgd=optimizer, losses=losses)

                except Exception as e:
                    pass

            print(losses)


train_model(train_data)

nlp.to_disk('nlp_model')

nlp_model = spacy.load('nlp_model')

# doc = nlp_model(train_data[2][0])
# for ent in doc.ents:
#   print(f'{ent.label_.upper():{30}}- {ent.text}')


fname = 'Alice Clark CV.pdf'
doc = fitz.open(fname)
text = ""
for page in doc:
    text = text + str(page.get_text())
tx = " ".join(text.split('\n'))
#print(tx)

doc = nlp_model(tx)
for ent in doc.ents:
    print(f'{ent.label_.upper():{30}}- {ent.text}')
