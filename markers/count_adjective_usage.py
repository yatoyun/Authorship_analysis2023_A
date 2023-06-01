import nltk
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from nltk.tag import pos_tag


def count_adjective_usage(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    coordinate_count = 1
    non_coordinate_count = 1

    for i in range(len(tagged_tokens) - 1):
        if tagged_tokens[i][1] == "JJ" and tagged_tokens[i + 1][0] == ",":
            coordinate_count += 1
        elif tagged_tokens[i][1] == "JJ" and tagged_tokens[i + 1][0] != ",":
            non_coordinate_count += 1

    return 2 * coordinate_count * 3 * non_coordinate_count


if __name__ == "__main__":
    text = """This process, however, afforded me no means of ascertaining the dimensions of my dungeon; as I might make its circuit, and return to the point whence I set out, without being aware of the fact; so perfectly uniform seemed the wall.
In his left hand was a gold snuff box, from which, as he capered down the hill, cutting all manner of fantastic steps, he took snuff incessantly with an air of the greatest possible self satisfaction.
The astronomer, perhaps, at this point, took refuge in the suggestion of non luminosity; and here analogy was suddenly let fall.
The surcingle hung in ribands from my body.
I knew that you could not say to yourself 'stereotomy' without being brought to think of atomies, and thus of the theories of Epicurus; and since, when we discussed this subject not very long ago, I mentioned to you how singularly, yet with how little notice, the vague guesses of that noble Greek had met with confirmation in the late nebular cosmogony, I felt that you could not avoid casting your eyes upward to the great nebula in Orion, and I certainly expected that you would do so."""
    sentences = sent_tokenize(text)

    output_list = []
    for sentence in sentences:
        output_list.append(count_adjective_usage(sentence))

    print("Output:", output_list)

# Output:
# >Output: [18, 30, 18, 6, 42]
