import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag


def attributive_and_predicative_adjective_value(sentence):
    classification_adjective = 1  # 0 is not used adjective, 1 = attribute, -1 = predicate
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)

    for i, (word, pos) in enumerate(tagged_words):
        if pos == "JJ":
            if i + 1 < len(tagged_words) and tagged_words[i + 1][1] == "NN":
                classification_adjective *= 2
            elif i > 0 and tagged_words[i - 1][1] in {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}:
                classification_adjective *= 3

    return classification_adjective


if __name__ == "__main__":
    text = """This process, however, afforded me no means of ascertaining the dimensions of my dungeon; as I might make its circuit, and return to the point whence I set out, without being aware of the fact; so perfectly uniform seemed the wall.
In his left hand was a gold snuff box, from which, as he capered down the hill, cutting all manner of fantastic steps, he took snuff incessantly with an air of the greatest possible self satisfaction.
The astronomer, perhaps, at this point, took refuge in the suggestion of non luminosity; and here analogy was suddenly let fall.
The surcingle hung in ribands from my body.
I knew that you could not say to yourself 'stereotomy' without being brought to think of atomies, and thus of the theories of Epicurus; and since, when we discussed this subject not very long ago, I mentioned to you how singularly, yet with how little notice, the vague guesses of that noble Greek had met with confirmation in the late nebular cosmogony, I felt that you could not avoid casting your eyes upward to the great nebula in Orion, and I certainly expected that you would do so."""
    sentences = sent_tokenize(text)

    output_list = []
    for sentence in sentences:
        output_list.append(attributive_and_predicative_adjective_value(sentence))

    print("Output:", output_list)

# Output: [3, 8, 4, 1, 8]
