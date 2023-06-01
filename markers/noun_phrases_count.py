import nltk
from nltk import pos_tag
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
    text = """
    The US has "passed the peak" on new coronavirus cases, 
    President Donald Trump said and predicted that some states would reopen this month."""
    print(count_noun_phrases(text))

    # output
    # >3
