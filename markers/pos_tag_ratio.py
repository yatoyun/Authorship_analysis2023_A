import nltk
from collections import Counter
from nltk import sent_tokenize
import numpy as np
from softmax import softmax

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


def pos_ratios(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    pos_counts = Counter(tag for word, tag in pos_tags)
    len_pos_counts = len(pos_counts)

    pos_ratios = [count for tag, count in pos_counts.items()]

    return np.prod(pos_ratios[:5])


if __name__ == "__main__":
    text = """This process, however, afforded me no means of ascertaining the dimensions of my dungeon; as I might make its circuit, and return to the point whence I set out, without being aware of the fact; so perfectly uniform seemed the wall.
In his left hand was a gold snuff box, from which, as he capered down the hill, cutting all manner of fantastic steps, he took snuff incessantly with an air of the greatest possible self satisfaction.
The astronomer, perhaps, at this point, took refuge in the suggestion of non luminosity; and here analogy was suddenly let fall.
The surcingle hung in ribands from my body.
I knew that you could not say to yourself 'stereotomy' without being brought to think of atomies, and thus of the theories of Epicurus; and since, when we discussed this subject not very long ago, I mentioned to you how singularly, yet with how little notice, the vague guesses of that noble Greek had met with confirmation in the late nebular cosmogony, I felt that you could not avoid casting your eyes upward to the great nebula in Orion, and I certainly expected that you would do so."""
    sentences = sent_tokenize(text)

    output_list = []
    for sentence in sentences:
        output_list.append(pos_ratios(sentence))

    print("Output:", output_list)

# Output: [1152, 648, 567, 6, 37440]
