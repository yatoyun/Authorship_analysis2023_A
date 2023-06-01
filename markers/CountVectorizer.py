from sklearn.feature_extraction.text import CountVectorizer
from nltk import sent_tokenize


text_example = """This process, however, afforded me no means of ascertaining the dimensions of my dungeon; as I might make its circuit, and return to the point whence I set out, without being aware of the fact; so perfectly uniform seemed the wall.
In his left hand was a gold snuff box, from which, as he capered down the hill, cutting all manner of fantastic steps, he took snuff incessantly with an air of the greatest possible self satisfaction.
The astronomer, perhaps, at this point, took refuge in the suggestion of non luminosity; and here analogy was suddenly let fall.
The surcingle hung in ribands from my body.
I knew that you could not say to yourself 'stereotomy' without being brought to think of atomies, and thus of the theories of Epicurus; and since, when we discussed this subject not very long ago, I mentioned to you how singularly, yet with how little notice, the vague guesses of that noble Greek had met with confirmation in the late nebular cosmogony, I felt that you could not avoid casting your eyes upward to the great nebula in Orion, and I certainly expected that you would do so."""

sentences = sent_tokenize(text_example)

tfidf_vec = CountVectorizer(stop_words="english", ngram_range=(1, 3))
train_tfidf = tfidf_vec.fit(sentences)
train_tfidf = tfidf_vec.transform(sentences)


print("Output:", train_tfidf)

"""
Output:   (0, 0)        1
  (0, 1)        1
  (0, 2)        1
  (0, 12)       1
  (0, 13)       1
  (0, 14)       1
  (0, 24)       1
  (0, 25)       1
  (0, 26)       1
  (0, 42)       1
  (0, 43)       1
  (0, 44)       1
  (0, 54)       1
  (0, 55)       1
  (0, 56)       1
  (0, 60)       1
  (0, 61)       1
  (0, 62)       1
  (0, 70)       1
  (0, 71)       1
  (0, 72)       1
  (0, 127)      1
  (0, 128)      1
  (0, 129)      1
  (0, 133)      1
  :     :
  (4, 159)      1
  (4, 183)      1
  (4, 184)      1
  (4, 185)      1
  (4, 191)      1
  (4, 192)      1
  (4, 193)      1
  (4, 202)      1
  (4, 203)      1
  (4, 204)      1
  (4, 205)      1
  (4, 206)      1
  (4, 207)      1
  (4, 217)      1
  (4, 218)      1
  (4, 219)      1
  (4, 220)      1
  (4, 221)      1
  (4, 222)      1
  (4, 230)      1
  (4, 231)      1
  (4, 232)      1
  (4, 233)      1
  (4, 234)      1
  (4, 235)      1"""
