### Using a Tagger
import  nltk

text = nltk.word_tokenize("And now for something completely different")
nltk.pos_tag(text)

text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
text.similar('woman')
text.similar('bought')
text.similar('over')
text.similar('the')

### Tagged Corpora
tagged_token = nltk.tag.str2tuple('fly/NN')
tagged_token
sent = '''
The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN
other/AP topics/NNS ,/, AMONG/IN them/PPO the/AT Atlanta/NP and/CC
Fulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PPS
said/VBD ``/`` ARE/BER well/QL operated/VBN and/CC follow/VB generally/RB
accepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT best/JJT
interest/NN of/IN both/ABX governments/NNS ''/'' ./.
'''
[nltk.tag.str2tuple(t) for t in sent.split()]

nltk.corpus.brown.tagged_words()
nltk.corpus.brown.tagged_words(simplify_tags=True)
nltk.corpus.nps_chat.tagged_words()
nltk.corpus.conll2000.tagged_words()
nltk.corpus.treebank.tagged_words()
nltk.corpus.brown.tagged_words(simplify_tags=True)
nltk.corpus.treebank.tagged_words(simplify_tags=True)
# other language
nltk.corpus.sinica_treebank.tagged_words()
nltk.corpus.indian.tagged_words()
nltk.corpus.mac_morpho.tagged_words()
nltk.corpus.conll2002.tagged_words()
nltk.corpus.cess_cat.tagged_words()

#############  tagset  #############
# ADJ         # adjective new, good, high, special, big, local
# ADV         # adverb really, already, still, early, now
# CNJ         # conjunction and, or, but, if, while, although
# DET         # determiner the, a, some, most, every, no
# EX          # existential there, there’s
# FW          # foreign word dolce, ersatz, esprit, quo, maitre
# MOD         # modal verb will, can, would, may, must, should
# N           # noun year, home, costs, time, education
# NP          # proper noun Alison, Africa, April, Washington
# NUM         # number twenty-four, fourth, 1991, 14:24
# PRO         # pronoun he, their, her, its, my, I, us
# P           # preposition on, of, at, with, by, into, under
# TO          # the word to to
# UH          # interjection ah, bang, ha, whee, hmpf, oops
# V           # verb is, has, get, do, make, see, run
# VD          # past tense said, took, told, made, asked
# VG          # present participle making, going, playing, working
# VN          # past participle given, taken, begun, sung
# WH          # wh determiner who, which, when, what, where, how

### Exploring Tagged Corpora
from nltk.corpus import brown
brown_learned_text = brown.words(categories='learned')
sorted(set(b for (a, b) in nltk.bigrams(brown_learned_text) if a == 'often'))

brown_lrnd_tagged = brown.tagged_words(categories='learned', tagset='universal')
tags = [b[1] for (a, b) in nltk.bigrams(brown_lrnd_tagged) if a[0] == 'often']
fd = nltk.FreqDist(tags)
fd.tabulate()

def process(sentence):
    for (w1,t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence):
        if (t1.startswith('V') and t2 == 'TO' and t3.startswith('V')):
            print(w1, w2, w3)

for tagged_sent in brown.tagged_sents():
    process(tagged_sent)

### Default Dictionaries
frequency = nltk.defaultdict(int)
frequency['colorless'] = 4
frequency['ideas']
pos = nltk.defaultdict(list)
pos['sleep'] = ['N', 'V']
pos['ideas']
pos = nltk.defaultdict(lambda: 'N')
pos['colorless'] = 'ADJ'
pos['blog']
pos.items()

alice = nltk.corpus.gutenberg.words('carroll-alice.txt')
vocab = nltk.FreqDist(alice)
v1000 = list(vocab)[:1000]
mapping = nltk.defaultdict(lambda: 'UNK')

### Incrementally Updating a Dictionary
counts = nltk.defaultdict(int)
for (word, tag) in brown.tagged_words(categories='news'):
    counts[tag] += 1
counts['N']
list(counts)

from operator import itemgetter
sorted(counts.items(), key=itemgetter(1), reverse=True)
[t for t, c in sorted(counts.items(), key=itemgetter(1), reverse=True)]

### Complex Keys and Values
pos = nltk.defaultdict(lambda: nltk.defaultdict(int))
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
for ((w1, t1), (w2, t2)) in nltk.bigrams(brown_news_tagged):
    pos[(t1, w2)][t2] += 1
pos[('DET', 'right')]

### Inverting a Dictionary
counts = nltk.defaultdict(int)
for word in nltk.corpus.gutenberg.words('milton-paradise.txt'):
    counts[word] += 1
[key for (key, value) in counts.items() if value == 32]

pos = {'colorless': 'ADJ', 'ideas': 'N', 'sleep': 'V', 'furiously': 'ADV'}
pos2 = dict((value, key) for (key, value) in pos.items())
pos2['N']

pos.update({'cats': 'N', 'scratch': 'V', 'peacefully': 'ADV', 'old': 'ADJ'})
pos2 = nltk.defaultdict(list)
for key, value in pos.items():
    pos2[value].append(key)
pos2['ADV']

pos2 = nltk.Index((value, key) for (key, value) in pos.items())
pos2['ADV']

############# dict ############
# d = {}              # Create an empty dictionary and assign it to d
# d[key] =            # value Assign a value to a given dictionary key
# d.keys()            # The list of keys of the dictionary
# list(d)             # The list of keys of the dictionary
# sorted(d)           # The keys of the dictionary, sorted
# key in d            # Test whether a particular key is in the dictionary
# for key in d        # Iterate over the keys of the dictionary
# d.values()          # The list of values in the dictionary
# dict([(k1,v1), (k2,v2), ...])   # Create a dictionary from a list of key-value pairs
# d1.update(d2)       # Add all items from d2 to d1
# defaultdict(int)    # A dictionary whose default value is zero

### The Default Tagger
tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
nltk.FreqDist(tags).max()
raw = 'I do not like green eggs and ham, I do not like them Sam I am!'
tokens = nltk.word_tokenize(raw)
default_tagger = nltk.DefaultTagger('NN')
default_tagger.tag(tokens)
default_tagger.evaluate(brown_tagged_sents)

### The Regular Expression Tagger
patterns = [(r'.*ing$', 'VBG'), # gerunds
            (r'.*ed$', 'VBD'), # simple past
            (r'.*es$', 'VBZ'), # 3rd singular present
            (r'.*ould$', 'MD'), # modals
            (r'.*\'s$', 'NN$'), # possessive nouns
            (r'.*s$', 'NNS'), # plural nouns
            (r'^-?[0-9]+(.[0-9]+)?$', 'CD'), # cardinal numbers
            (r'.*', 'NN') # nouns (default)
]
regexp_tagger = nltk.RegexpTagger(patterns)
regexp_tagger.tag(brown_sents[3])
regexp_tagger.evaluate(brown_tagged_sents)

### The Lookup Tagger
fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = fd.keys()[:100]
likely_tags = dict((word, cfd[word].max()) for word in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
baseline_tagger.evaluate(brown_tagged_sents)

sent = brown.sents(categories='news')[3]
baseline_tagger.tag(sent)
baseline_tagger = nltk.UnigramTagger(model=likely_tags,
                                     backoff=nltk.DefaultTagger('NN'))
def performance(cfd, wordlist):
    lt = dict((word, cfd[word].max()) for word in wordlist)
    baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger('NN'))
    return baseline_tagger.evaluate(brown.tagged_sents(categories='news'))

def display():
    import pylab
    words_by_freq = list(nltk.FreqDist(brown.words(categories='news')))
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    sizes = 2 ** pylab.arange(15)
    perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]
    pylab.plot(sizes, perfs, '-bo')
    pylab.title('Lookup Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()