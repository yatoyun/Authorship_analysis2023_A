def count_noun_phrases(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Perform POS tagging
    tagged_tokens = pos_tag(tokens)

    # Define grammar for noun phrases
    np_grammar = "NP: {<DT>?<JJ>*<NN>}"

    # Create parser
    parser = RegexpParser(np_grammar)

    # Parse the tagged tokens
    tree = parser.parse(tagged_tokens)

    # Initialize NP count
    np_count = 0

    # Iterate through the tree and count all NPs
    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            np_count += 1

    return np_count
