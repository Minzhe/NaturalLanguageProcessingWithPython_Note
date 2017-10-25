### Gutenberg Corpus
import nltk
nltk.corpus.gutenberg.fileids()
emma = nltk.corpus.gutenberg.words('austen-emma.txt')
len(emma)
emma = nltk.Text(emma)
emma.concordance('surprize')

from nltk.corpus import gutenberg
for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set([w.lower() for w in gutenberg.words(fileid)]))
    print(int(num_chars/num_words), int(num_words/num_sents), int(num_words/num_vocab), fileid)

macbeth_sents = gutenberg.sents('shakespeare-macbeth.txt')
macbeth_sents
macbeth_sents[1037]
longest_len = max([len(s) for s in macbeth_sents])
[s for s in macbeth_sents if len(s) == longest_len]

### web and chat text
from nltk.corpus import webtext
for fileid in webtext.fileids():
    print(fileid, webtext.raw(fileid)[:65], '...')

### Brown Corpus
from nltk.corpus import brown
brown.categories()
brown.words(categories='news')
brown.words(fileids=['cg22'])
brown.sents(categories=['news', 'editorial', 'review'])

news_text = brown.words(categories='news')
fdist = nltk.FreqDist([w.lower() for w in news_text])
modals = ['can', 'could', 'may', 'might', 'must', 'will']
for m in modals:
    print(m + ':', fdist[m])

cfd = nltk.ConditionalFreqDist((genre, word)
                               for genre in brown.categories()
                               for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfd.tabulate(conditons=genres, samples=modals)

### Reuters Corpus
from nltk.corpus import reuters
reuters.fileids()
reuters.categories()
reuters.categories('training/9865')
reuters.categories(['training/9865', 'training/9880'])
reuters.fileids('barley')
reuters.fileids(['barley', 'corn'])
reuters.words('training/9865')[:14]
reuters.words(['training/9865', 'training/9880'])
reuters.words(categories='barley')
reuters.words(categories=['barley', 'corn'])

### inaugural address corpus
from nltk.corpus import inaugural
inaugural.fileids()
[fileid[:4] for fileid in inaugural.fileids()]
cfd = nltk.ConditionalFreqDist((target, fileid[:4])
                               for fileid in inaugural.fileids()
                               for w in inaugural.words(fileid)
                               for target in ['america', 'citizen']
                               if w.lower().startswith(target)
)
cfd.plot()

### other language
nltk.corpus.cess_esp.words()
nltk.corpus.floresta.words()
nltk.corpus.indian.words('hindi.pos')
nltk.corpus.udhr.fileids()
nltk.corpus.udhr.words('Javanese-Latin1')[11:]

from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist((lang, len(word))
                               for lang in languages
                               for word in udhr.words(lang + '-Latin1'))
cfd.plot(cumulative=True)
cfd.tabulate(conditions=['English', 'German_Deutsch'], samples=range(10), cumulative=True)

##############  structure  #####################
# fileids()                           # The files of the corpus
# fileids([categories])               # The files of the corpus corresponding to these categories
# categories()                        # The categories of the corpus
# categories([fileids])               # The categories of the corpus corresponding to these files
# raw()                               # The raw content of the corpus
# raw(fileids=[f1,f2,f3])             # The raw content of the specified files
# raw(categories=[c1,c2])             # The raw content of the specified categories
# words()                             # The words of the whole corpus
# words(fileids=[f1,f2,f3])           # The words of the specified fileids
# words(categories=[c1,c2])           # The words of the specified categories
# sents()                             # The sentences of the specified categories
# sents(fileids=[f1,f2,f3])           # The sentences of the specified fileids
# sents(categories=[c1,c2])           # The sentences of the specified categories
# abspath(fileid)                     # The location of the given file on disk
# encoding(fileid)                    # The encoding of the file (if known)
# open(fileid)                        # Open a stream for reading the given corpus file
# root()                              # The path to the root of locally installed corpus
# readme()                            # The contents of the README file of the corpus

raw = gutenberg.raw("burgess-busterbrown.txt")
raw[1:20]
words = gutenberg.words("burgess-busterbrown.txt")
words[1:20]
sents = gutenberg.sents("burgess-busterbrown.txt")
sents[1:20]

### own corpus
from nltk.corpus import PlaintextCorpusReader
corpus_root = '/usr/share/dict'
wordlists = PlaintextCorpusReader(corpus_root, '.*')
wordlists.fileids()
wordlists.words('words')

### Conditional Frequency Distributions
from nltk.corpus import brown
cfd = nltk.ConditionalFreqDist((genre, word)
                               for genre in brown.categories()
                               for word in brown.words(categories=genre))
genre_word = [(genre, word)
              for genre in ['news', 'romance']
              for word in brown.words(categories=genre)]
len(genre_word)
genre_word[:4]
genre_word[-4:]

cfd = nltk.ConditionalFreqDist(genre_word)
cfd
cfd.conditions()
cfd['news']
cfd['romance']
list(cfd['romance'])
cfd['romance']['could']

from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist((lang, len(word))
                               for lang in languages
                               for word in udhr.words(lang + '-Latin1'))
cfd.tabulate(conditions=['English', 'German_Deutsch'], samples=range(10), cumulative=True)

### random text
def generate_model(cfdist, word, num=15):
    for i in range(num):
        print(word)
        word = cfdist[word].max()
text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)
cfd['living']
generate_model(cfd, 'living')

################# Conditional Frequency Distributions ###########################
# cfdist = ConditionalFreqDist(pairs)         # Create a conditional frequency distribution from a list of pairs
# cfdist.conditions()                         # Alphabetically sorted list of conditions
# cfdist[condition]                           # The frequency distribution for this condition
# cfdist[condition][sample]                   # Frequency for the given sample for this condition
# cfdist.tabulate()                           # Tabulate the conditional frequency distribution
# cfdist.tabulate(samples, conditions)        # Tabulation limited to the specified samples and conditions
# cfdist.plot()                               # Graphical plot of the conditional frequency distribution
# cfdist.plot(samples, conditions)            # Graphical plot limited to the specified samples and conditions
# cfdist1 < cfdist2                           # Test if samples in cfdist1 occur less frequently than in cfdist2


############ lexical ############
def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab.difference(english_vocab)
    return sorted(unusual)
unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt'))

from nltk.corpus import stopwords
stopwords.words('english')

def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)
content_fraction(nltk.corpus.reuters.words())

puzzle_letters = nltk.FreqDist('egivrvonl')
obligatory = 'r'
wordlist = nltk.corpus.words.words()
[w for w in wordlist if len(w) >= 6 and obligatory in w and nltk.FreqDist(w) <= puzzle_letters]

### name corpus
names = nltk.corpus.names
names.fileids()
male_names = names.words('male.txt')
female_names = names.words('female.txt')
[w for w in male_names if w in female_names]

cfd = nltk.ConditionalFreqDist((fileid, name[-1])
                               for fileid in names.fileids()
                               for name in names.words(fileid))
cfd.plot()

### Pronouncing Dictionary
entries = nltk.corpus.cmudict.entries()
len(entries)
for entry in entries[39943:39951]:
    print(entry)

### Comparative Wordlists
from nltk.corpus import swadesh
swadesh.fileids()
swadesh.words('en')
fr2en = swadesh.entries(['fr', 'en'])
fr2en
translate = dict(fr2en)
translate['chien']
translate['jeter']
de2en = swadesh.entries(['de', 'en'])  # German-English
es2en = swadesh.entries(['es', 'en'])  # Spanish-English
translate.update(dict(de2en))
translate.update(dict(es2en))
translate['Hund']
translate['perro']

languages = ['en', 'de', 'nl', 'es', 'fr', 'pt', 'la']
for i in [139, 140, 141, 142]:
    print(swadesh.entries(languages)[i])

### WordNet
from nltk.corpus import wordnet as wn
wn.synsets('motorcar')
wn.synset('car.n.01').lemma_names()
wn.synset('car.n.01').definition()
wn.synset('car.n.01').examples()
wn.synset('car.n.01').lemmas()
wn.lemma('car.n.01.automobile')
wn.lemma('car.n.01.automobile').synset()
wn.lemma('car.n.01.automobile').name()
wn.synsets('car')
for synset in wn.synsets('car'):
    print(synset.lemma_names())

### The WordNet Hierarchy
motorcar = wn.synset('car.n.01')
types_of_motorcar = motorcar.hyponyms()
types_of_motorcar[26]
motorcar.hypernyms()
paths = motorcar.hypernym_paths()
len(paths)
[synset.name for synset in paths[0]]
[synset.name for synset in paths[1]]
motorcar.root_hypernyms()

### More Lexical Relations
wn.synset('tree.n.01').part_meronyms()
wn.synset('tree.n.01').substance_meronyms()
wn.synset('tree.n.01').member_holonyms()
for synset in wn.synsets('mint', wn.NOUN):
    print(synset.name + ':', synset.definition)
wn.synset('mint.n.04').part_holonyms()
wn.synset('mint.n.04').substance_holonyms()
wn.synset('walk.v.01').entailments()
wn.synset('eat.v.01').entailments()
wn.synset('tease.v.03').entailments()
wn.lemma('supply.n.02.supply').antonyms()
wn.lemma('rush.v.01.rush').antonyms()
wn.lemma('horizontal.a.01.horizontal').antonyms()
wn.lemma('staccato.r.01.staccato').antonyms()

