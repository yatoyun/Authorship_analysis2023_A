from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import sent_tokenize


text_example = """This process, however, afforded me no means of ascertaining the dimensions of my dungeon; as I might make its circuit, and return to the point whence I set out, without being aware of the fact; so perfectly uniform seemed the wall.
In his left hand was a gold snuff box, from which, as he capered down the hill, cutting all manner of fantastic steps, he took snuff incessantly with an air of the greatest possible self satisfaction.
The astronomer, perhaps, at this point, took refuge in the suggestion of non luminosity; and here analogy was suddenly let fall.
The surcingle hung in ribands from my body.
I knew that you could not say to yourself 'stereotomy' without being brought to think of atomies, and thus of the theories of Epicurus; and since, when we discussed this subject not very long ago, I mentioned to you how singularly, yet with how little notice, the vague guesses of that noble Greek had met with confirmation in the late nebular cosmogony, I felt that you could not avoid casting your eyes upward to the great nebula in Orion, and I certainly expected that you would do so."""

sentences = sent_tokenize(text_example)

# main code
tfidf_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
train_tfidf = tfidf_vec.fit(sentences)
train_tfidf = tfidf_vec.transform(sentences)


print("Output:", train_tfidf)

# output
"""
Output:   (0, 236)      0.14965278840547694
  (0, 229)      0.14965278840547694
  (0, 228)      0.14965278840547694
  (0, 190)      0.14965278840547694
  (0, 189)      0.14965278840547694
  (0, 188)      0.14965278840547694
  (0, 179)      0.14965278840547694
  (0, 178)      0.14965278840547694
  (0, 177)      0.14965278840547694
  (0, 173)      0.14965278840547694
  (0, 172)      0.14965278840547694
  (0, 171)      0.14965278840547694
  (0, 165)      0.14965278840547694
  (0, 164)      0.14965278840547694
  (0, 163)      0.12073892739495884
  (0, 162)      0.14965278840547694
  (0, 161)      0.14965278840547694
  (0, 160)      0.14965278840547694
  (0, 135)      0.14965278840547694
  (0, 134)      0.14965278840547694
  (0, 133)      0.14965278840547694
  (0, 129)      0.14965278840547694
  (0, 128)      0.14965278840547694
  (0, 127)      0.14965278840547694
  (0, 72)       0.14965278840547694
  :     :
  (4, 58)       0.09901475429766735
  (4, 57)       0.09901475429766735
  (4, 50)       0.09901475429766735
  (4, 49)       0.09901475429766735
  (4, 48)       0.09901475429766735
  (4, 47)       0.09901475429766735
  (4, 46)       0.09901475429766735
  (4, 45)       0.09901475429766735
  (4, 41)       0.09901475429766735
  (4, 40)       0.09901475429766735
  (4, 39)       0.09901475429766735
  (4, 38)       0.09901475429766735
  (4, 37)       0.09901475429766735
  (4, 33)       0.09901475429766735
  (4, 32)       0.09901475429766735
  (4, 31)       0.09901475429766735
  (4, 23)       0.09901475429766735
  (4, 22)       0.09901475429766735
  (4, 21)       0.09901475429766735
  (4, 20)       0.09901475429766735
  (4, 19)       0.09901475429766735
  (4, 18)       0.09901475429766735
  (4, 5)        0.09901475429766735
  (4, 4)        0.09901475429766735
  (4, 3)        0.09901475429766735"""
