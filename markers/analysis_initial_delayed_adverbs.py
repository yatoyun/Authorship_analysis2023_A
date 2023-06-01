from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def analyze_adverbs(text):
    sentences = sent_tokenize(text)

    initial_adverbs = 1
    delayed_adverbs = 1

    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged = pos_tag(words)

        for i in range(len(tagged) - 1):
            if tagged[i][1] == 'RB' and tagged[i + 1][1] == ',':
                initial_adverbs += 1
            elif tagged[i][1] == ',' and tagged[i + 1][1] == 'RB':
                delayed_adverbs += 1
    return 2 * initial_adverbs * 3 * delayed_adverbs


if __name__ == "__main__":
    text = """He was an old man who fished alone in a skiff in the Gulf Stream and he had gone eighty-four days now without taking a fish.
In the first forty days a boy had been with him.
But after forty days without a fish the boy’s parents had told him that the old man was now definitely and finally salao, which is the worst form of unlucky, and the boy had gone at their orders in another boat which caught three good fish the first week.
It made the boy sad to see the old man come in each day with his skiff empty and he always went down to help him carry either the coiled lines or the gaff and harpoon and the sail that was furled around the mast.
The sail was patched with flour sacks and, furled, it looked like the flag of permanent defeat.
The old man was thin and gaunt with deep wrinkles in the back of his neck.
The brown blotches of the benevolent skin cancer the sun brings from its reflection on the tropic sea were on his cheeks.
"""
    sentences = sent_tokenize(text)
    output_list = []
    for sentence in sentences:
        output_list.append(analyze_adverbs(sentence))

    print("Output:", output_list)
