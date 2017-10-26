### Download
import nltk
nltk.download()
from nltk.book import *
text1

### search
text1.concordance('monstrous')
text1.similar('monstrous')
text2.similar('monstrous')
text2.common_contexts(['monstrous', 'very'])
# plot dispersion
text4.dispersion_plot(['citizens', 'democracy', 'freedom', 'duties', 'America'])
text3.generate() # not in nltk3

### Counting vocabulary
len(text3)
sorted(set(text3))
len(set(text3))
len(text3) / len(set(text3))
text3.count('smote')
100 * text4.count('a') / len(text4)

### List of words
text4.index('awaken')
text5[16715:16735]

### sample statistics
fdist1 = FreqDist(text1)
vocabulary1 = fdist1.keys()
fdist1['whale']
fdist1.plot(50, cumulative = True)

### word selection
V = set(text1)
long_words = [w for w in V if len(w) > 15]
sorted(long_words)

fdist5 = FreqDist(text5)
sorted([w for w in set(text5) if len(w) > 7 and fdist5[w] > 7])

### collocations and bigrams
list(nltk.bigrams(['more', 'is', 'said', 'than', 'done']))
text4.collocations()

### counting other things
fdist = FreqDist([len(w) for w in text1])
fdist.keys()
fdist.items()
fdist.max()
fdist.freq(3)

############ nltk frequency function ############
# fdist = FreqDist(samples)       # Create a frequency distribution containing the given samples
# fdist.inc(sample)               # Increment the count for this sample
# fdist['monstrous']              # Count of the number of times a given sample occurred
# fdist.freq('monstrous')         # Frequency of a given sample
# fdist.N()                       # Total number of samples
# fdist.keys()                    # The samples sorted in order of decreasing frequency
# for sample in fdist:            # Iterate over the samples, in order of decreasing frequency
# fdist.max()                     # Sample with the greatest count
# fdist.tabulate()                # Tabulate the frequency distribution
# fdist.plot()                    # Graphical plot of the frequency distribution
# fdist.plot(cumulative=True)     # Cumulative plot of the frequency distribution
# fdist1 < fdist2 Test if samples in fdist1   # occur less frequently than in fdist2

########## conditon ###########
# s.startswith(t)
# s.endswith(t)
# t in s
# s.islower()
# s.isupper()
# s.isalpha()
# s.isalnum()
# s.isdigit()
# s.istitle()

### machine translation
babelize_shell()









