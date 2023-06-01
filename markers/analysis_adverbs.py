from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def load_text(file):
    with open(file, 'r') as f:
        text = f.read()
    return text


def identify_markers(text):
    sentences = sent_tokenize(text)

    fronted_adverbs = 0
    regular_adverbs = 0

    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged = pos_tag(words)

        for i in range(len(tagged) - 1):
            if tagged[i][1] == 'RB' and tagged[i + 1][1].startswith('VB'):
                fronted_adverbs += 1
            elif tagged[i][1].startswith('VB') and tagged[i + 1][1] == 'RB':
                regular_adverbs += 1

    return fronted_adverbs, regular_adverbs


def analyze_text(file):
    text = load_text(file)
    fronted_adverbs, regular_adverbs = identify_markers(text)

    print(f"Fronted Adverbs of Manner: {fronted_adverbs}")
    print(f"Regular Adverbs of Manner: {regular_adverbs}")

    if regular_adverbs == 0:
        print("No regular adverbs of manner found.")
    else:
        print(
            f"Ratio of Fronted Adverbs to Regular Adverbs: {fronted_adverbs / regular_adverbs}")


analyze_text('a.txt')

"""
input_txt******************************************************************************************************************************************************************************************************************************************************
    He was an old man who fished alone in a skiff in the Gulf Stream and he had gone eighty-four days now without taking a fish.
In the first forty days a boy had been with him.
But after forty days without a fish the boy’s parents had told him that the old man was now definitely and finally salao, which is the worst form of unlucky, and the boy had gone at their orders in another boat which caught three good fish the first week.
It made the boy sad to see the old man come in each day with his skiff empty and he always went down to help him carry either the coiled lines or the gaff and harpoon and the sail that was furled around the mast.
The sail was patched with flour sacks and, furled, it looked like the flag of permanent defeat.
The old man was thin and gaunt with deep wrinkles in the back of his neck.
The brown blotches of the benevolent skin cancer the sun brings from its reflection on the tropic sea were on his cheeks.
***************************************************************************************************************************************************************************************************************************************************************
output>>>
Fronted Adverbs of Manner: 1
Regular Adverbs of Manner: 3
Ratio of Fronted Adverbs to Regular Adverbs: 0.3333333333333333
    """
