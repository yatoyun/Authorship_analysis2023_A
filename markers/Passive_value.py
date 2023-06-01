import nltk
from nltk import sent_tokenize

##reference : https://github.com/flycrane01/nltk-passive-voice-detector-for-English/blob/master/Passive-voice.py


def isPassive(sentence):
    beforms = ["is", "am", "are", "was", "were", "been", "be", "being", "'s", "'m", "'re"]  # all forms of "be"
    aux = [
        "do",
        "did",
        "does",
        "have",
        "has",
        "had",
    ]  # NLTK tags "do" and "have" as verbs, which can be misleading in the following section.
    words = nltk.word_tokenize(sentence)
    tokens = nltk.pos_tag(words)
    tags = [i[1] for i in tokens]
    if tags.count("VBN") == 0:  # no PP, no passive voice.
        return 0
    elif tags.count("VBN") == 1 and "been" in words:  # one PP "been", still no passive voice.
        return 0
    else:
        pos = [
            i for i in range(len(tags)) if tags[i] == "VBN" and words[i] != "been"
        ]  # gather all the PPs that are not "been".
        for end in pos:
            chunk = tags[:end]
            start = 0
            for i in range(len(chunk), 0, -1):
                last = chunk.pop()
                if last == "NN" or last == "PRP":
                    start = i  # get the chunk between PP and the previous NN or PRP (which in most cases are subjects)
                    break
            sentchunk = words[start:end]
            tagschunk = tags[start:end]
            verbspos = [
                i for i in range(len(tagschunk)) if tagschunk[i].startswith("V")
            ]  # get all the verbs in between
            num_beforms = 1
            num_aux = 1
            if verbspos != []:  # if there are no verbs in between, it's not passive
                for i in verbspos:
                    if sentchunk[i].lower() in beforms:
                        num_beforms += 1
                    elif sentchunk[i].lower() in aux:
                        num_aux += 1
                    else:  # check if they are all forms of "be" or auxiliaries such as "do" or "have".
                        break
                else:
                    return (2 * num_beforms) * (3 * num_aux)
    return 0


if __name__ == "__main__":
    text = """This process, however, afforded me no means of ascertaining the dimensions of my dungeon; as I might make its circuit, and return to the point whence I set out, without being aware of the fact; so perfectly uniform seemed the wall.
In his left hand was a gold snuff box, from which, as he capered down the hill, cutting all manner of fantastic steps, he took snuff incessantly with an air of the greatest possible self satisfaction.
The astronomer, perhaps, at this point, took refuge in the suggestion of non luminosity; and here analogy was suddenly let fall.
The surcingle hung in ribands from my body.
I knew that you could not say to yourself 'stereotomy' without being brought to think of atomies, and thus of the theories of Epicurus; and since, when we discussed this subject not very long ago, I mentioned to you how singularly, yet with how little notice, the vague guesses of that noble Greek had met with confirmation in the late nebular cosmogony, I felt that you could not avoid casting your eyes upward to the great nebula in Orion, and I certainly expected that you would do so."""
    sentences = sent_tokenize(text)

    output_list = []
    for sentence in sentences:
        output_list.append(isPassive(sentence))

    print("Output:", output_list)

# Output: [0, 0, 0, 0, 12]
