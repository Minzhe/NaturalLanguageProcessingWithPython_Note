### Accessing Text from the Web and from Disk
import nltk
from urllib.request import urlopen
import re

url = "http://www.gutenberg.org/files/2554/2554.txt"
raw = urlopen(url).read()
type(raw)
len(raw)
raw[:75]

tokens = nltk.word_tokenize(raw)
type(tokens)
len(tokens)
tokens[:10]

text = nltk.Text(tokens)
type(text)
text[1020:1060]
text.collocations()

raw.find("PART I")
raw.rfind("End of Project Gutenberg's Crime")
raw = raw[5303:1157681]
raw.find("PART I")

### Dealing with HTML
url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"
html = urlopen(url).read()
html[:60]
raw = nltk.clean_html(html)
tokens = nltk.word_tokenize(raw)
tokens
tokens = tokens[96:399]
text = nltk.Text(tokens)
text.concordance('gene')


################  String Operations  ############
# s.find(t)           # Index of first instance of string t inside s (-1 if not found)
# s.rfind(t)          # Index of last instance of string t inside s (-1 if not found)
# s.index(t)          # Like s.find(t), except it raises ValueError if not found
# s.rindex(t)         # Like s.rfind(t), except it raises ValueError if not found
# s.join(text)        # Combine the words of the text into a string using s as the glue
# s.split(t)          # Split s into a list wherever a t is found (whitespace by default)
# s.splitlines()      # Split s into a list of strings, one per line
# s.lower()           # A lowercased version of the string s
# s.upper()           # An uppercased version of the string s
# s.titlecase()       # A titlecased version of the string s
# s.strip()           # A copy of s without leading or trailing whitespace
# s.replace(t, u)     # Replace instances of t with u inside s


############## regular expression ############
# .       # Wildcard, matches any character
# ^abc    # Matches some pattern abc at the start of a string
# abc$    # Matches some pattern abc at the end of a string
# [abc]   # Matches one of a set of characters
# [A-Z0-9]    # Matches one of a range of characters
# ed|ing|s    # Matches one of the specified strings (disjunction)
# *       # Zero or more of previous item, e.g., a*, [a-z]* (also known as Kleene Closure)
# +       # One or more of previous item, e.g., a+, [a-z]+
# ?       # Zero or one of the previous item (i.e., optional), e.g., a?, [a-z]?
# {n}     # Exactly n repeats where n is a non-negative integer
# {n,}    # At least n repeats
# {,n}    # No more than n repeats
# {m,n}   # At least m and no more than n repeats
# a(b|c)+     # Parentheses that indicate the scope of the operators

### Searching Tokenized Text
from nltk.corpus import gutenberg, nps_chat
moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
moby.findall(r"<a> (<.*>) <man>")
chat = nltk.Text(nps_chat.words())
chat.findall(r"<.*> <.*> <bro>")

### Normalizing Text
raw = """DENNIS: Listen, strange women lying in ponds distributing swords
is no basis for a system of government. Supreme executive power derives from
a mandate from the masses, not from some farcical aquatic ceremony."""
tokens = nltk.word_tokenize(raw)
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
[porter.stem(t) for t in tokens]
[lancaster.stem(t) for t in tokens]


class IndexedText(object):
    def __init__(self, stemmer, text):
        self._text = text
        self._stemmer = stemmer
        self._index = nltk.Index((self._stem(word), i)
                                 for (i, word) in enumerate(text))

def concordance(self, word, width=40):
    key = self._stem(word)
    wc = width/4 # words of context
    for i in self._index[key]:
        lcontext = ' '.join(self._text[i-wc:i])
        rcontext = ' '.join(self._text[i:i+wc])
        ldisplay = '%*s' % (width, lcontext[-width:])
        rdisplay = '%-*s' % (width, rcontext[:width])
        print(ldisplay, rdisplay)

def _stem(self, word):
    return self._stemmer.stem(word).lower()

porter = nltk.PorterStemmer()
grail = nltk.corpus.webtext.words('grail.txt')
text = IndexedText(porter, grail)
text.concordance('lie')

### Lemmatization
wnl = nltk.WordNetLemmatizer()
[wnl.lemmatize(t) for t in tokens]

### Simple Approaches to Tokenization
raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful tone
though), 'I won't have any pepper in my kitchen AT ALL. Soup does very
well without--Maybe it's always pepper that makes people hot-tempered,'..."""
re.split(r' ', raw)
re.split(r'[ \t\n]+', raw)
re.split(r'\s+', raw)
re.split(r'\W+', raw)
re.findall(r'\w+|\S\w*', raw)
re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", raw)

# Symbol Function
# \b      # Word boundary (zero width)
# \d      # Any decimal digit (equivalent to [0-9])
# \D      # Any non-digit character (equivalent to [^0-9])
# \s      # Any whitespace character (equivalent to [ \t\n\r\f\v]
# \S      # Any non-whitespace character (equivalent to [^ \t\n\r\f\v])
# \w      # Any alphanumeric character (equivalent to [a-zA-Z0-9_])
# \W      # Any non-alphanumeric character (equivalent to [^a-zA-Z0-9_])
# \t      # The tab character
# \n      # The newline character

### NLTK’s Regular Expression Tokenizer
text = 'That U.S.A. poster-print costs $12.40...'
pattern = r'''(?x) # set flag to allow verbose regexps
([A-Z]\.)+ # abbreviations, e.g. U.S.A.
| \w+(-\w+)* # words with optional internal hyphens
| \$?\d+(\.\d+)?%? # currency and percentages, e.g. $12.40, 82%
| \.\.\. # ellipsis
| [][.,;"'?():-_`] # these are separate tokens
'''
nltk.regexp_tokenize(text, pattern)

### Sentence Segmentation
len(nltk.corpus.brown.words()) / len(nltk.corpus.brown.sents())
def segment(text, segs):
    words = []
    last = 0
    for i in range(len(segs)):
        if segs[i] == '1':
            words.append(text[last:i+1])
            last = i+1
    words.append(text[last:])
    return words
text = "doyouseethekittyseethedoggydoyoulikethekittylikethedoggy"
seg1 = "0000000000000001000000000010000000000000000100000000000"
seg2 = "0100100100100001001001000010100100010010000100010010000"
segment(text, seg1)

### Lining Things Up
'%6s' % 'dog'
'%-6s' % 'dog'
width = 6
'%-*s' % (width, 'dog')
count, total = 3205, 9375
"accuracy for %d words: %2.4f%%" % (total, 100 * count / total)

def tabulate(cfdist, words, categories):
    print('%-16s' % 'Category')
    for word in words: # column headings
        print('%6s' % word)
        # print()
    for category in categories:
        print('%-16s' % category) # row heading
        for word in words: # for each word
            print('%6d' % cfdist[category][word]) # print table cell
        # print() # end the row
from nltk.corpus import brown
cfd = nltk.ConditionalFreqDist((genre, word)
                               for genre in brown.categories()
                               for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
tabulate(cfd, modals, genres)