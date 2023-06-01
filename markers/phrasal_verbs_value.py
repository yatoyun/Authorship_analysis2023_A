import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize


def detect_phrasal_verbs(text):
    tokens = word_tokenize(text)

    # pos tag
    tagged = pos_tag(tokens)

    phrasal_verbs_vap = 1  # verb + adverb + preposition
    phrasal_verbs_vaop = 1  # verb + adverb or preposition
    for i in range(len(tagged) - 2):
        # Check verb or not
        if "VB" in tagged[i][1]:
            # Check next word is an adverb or preposition
            if "RB" in tagged[i + 1][1] or "IN" in tagged[i + 1][1]:
                # Check after next is a preposition
                if "IN" in tagged[i + 2][1]:
                    # If the verb + adverb + preposition pattern is met
                    phrasal_verbs_vap += 1
                else:
                    # If the verb + adverb or preposition pattern is met
                    phrasal_verbs_vaop += 1

    return 2 * phrasal_verbs_vap * 3 * phrasal_verbs_vaop


if __name__ == "__main__":
    text = """This process, however, afforded me no means of ascertaining the dimensions of my dungeon; as I might make its circuit, and return to the point whence I set out, without being aware of the fact; so perfectly uniform seemed the wall.
In his left hand was a gold snuff box, from which, as he capered down the hill, cutting all manner of fantastic steps, he took snuff incessantly with an air of the greatest possible self satisfaction.
The astronomer, perhaps, at this point, took refuge in the suggestion of non luminosity; and here analogy was suddenly let fall.
The surcingle hung in ribands from my body.
I knew that you could not say to yourself 'stereotomy' without being brought to think of atomies, and thus of the theories of Epicurus; and since, when we discussed this subject not very long ago, I mentioned to you how singularly, yet with how little notice, the vague guesses of that noble Greek had met with confirmation in the late nebular cosmogony, I felt that you could not avoid casting your eyes upward to the great nebula in Orion, and I certainly expected that you would do so."""
    sentences = sent_tokenize(text)

    output_list = []
    for sentence in sentences:
        output_list.append(detect_phrasal_verbs(sentence))

    print("Output:", output_list)
    # output
    # >Output: [6, 6, 12, 6, 42]
