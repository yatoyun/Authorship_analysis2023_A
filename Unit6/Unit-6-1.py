import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag


def analyze_adjective_usage(text):
    sentences = sent_tokenize(text)

    attributive_count = 0
    predicative_count = 0

    sentence_identification_dict = {}  # 1 for attributive, 2 for predicative, 0 for not identified

    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)

        for i, (word, pos) in enumerate(tagged_words):
            if pos == "JJ":
                if i + 1 < len(tagged_words) and tagged_words[i + 1][1] == "NN":
                    attributive_count += 1
                    sentence_identification_dict[sentence] = 0
                elif i > 0 and tagged_words[i - 1][1] in {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}:
                    predicative_count += 1
                    sentence_identification_dict[sentence] = 1
    ratio = (attributive_count, predicative_count)
    return ratio, sentence_identification_dict


text = input("Enter text: ")
result, result_dict = analyze_adjective_usage(text)
print("Sentence identification:")
for sentence, value in result_dict.items():
    print(f"{value} : {sentence}")
print(f"Attributive adjectives: {result[0]}, Predicative adjectives: {result[1]}")
