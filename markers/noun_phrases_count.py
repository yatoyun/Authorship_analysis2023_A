import nltk
from nltk import pos_tag, sent_tokenize
from nltk import RegexpParser


def count_noun_phrases(text):
    tokens = nltk.word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    # Define noun phrase chunk grammar
    np_grammar = "NP: {<DT>?<JJ>*<NN>}"

    # Create parser
    parser = RegexpParser(np_grammar)

    # Parse tagged tokens
    tree = parser.parse(tagged_tokens)

    # count noun phrases
    np_count = 0
    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            np_count += 1

    return np_count


if __name__ == "__main__":
    text = """This process, however, afforded me no means of ascertaining the dimensions of my dungeon; as I might make its circuit, and return to the point whence I set out, without being aware of the fact; so perfectly uniform seemed the wall.
In his left hand was a gold snuff box, from which, as he capered down the hill, cutting all manner of fantastic steps, he took snuff incessantly with an air of the greatest possible self satisfaction.
The astronomer, perhaps, at this point, took refuge in the suggestion of non luminosity; and here analogy was suddenly let fall.
The surcingle hung in ribands from my body.
I knew that you could not say to yourself 'stereotomy' without being brought to think of atomies, and thus of the theories of Epicurus; and since, when we discussed this subject not very long ago, I mentioned to you how singularly, yet with how little notice, the vague guesses of that noble Greek had met with confirmation in the late nebular cosmogony, I felt that you could not avoid casting your eyes upward to the great nebula in Orion, and I certainly expected that you would do so."""
    sentences = sent_tokenize(text)

    output_list = []
    for sentence in sentences:
        output_list.append(count_noun_phrases(sentence))

    print("Output:", output_list)
    # output
    # >Output: [8, 9, 7, 3, 5]
