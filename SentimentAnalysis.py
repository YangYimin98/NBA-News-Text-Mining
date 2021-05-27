from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import math
import numpy as np
from textblob import TextBlob
from sklearn.metrics import classification_report

snowballstemmer = SnowballStemmer("english")
stop_words = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')
word_net_lemmatizer = WordNetLemmatizer()
# nltk.download('vader_lexicon')


sns.set(style='darkgrid', context='talk', palette='Dark2')
sia = SIA()
results = []


def process_text(sent):
    tokens = []
    for word in sent:
        tok = tokenizer.tokenize(word)
        tok = [t.lower() for t in tok if t.lower() not in stop_words]
        tokens.extend(tok)

    return tokens


"""loading the dataset"""
f1 = open("staff.txt", 'r', encoding='UTF-8')


for line in f1:
    pol_score = sia.polarity_scores(line)
    pol_score['sentence'] = line
    results.append(pol_score)
# from pprint import pprint
# pprint(results[:3], width=100)

"""create a positive label of 1 if the compound> 0.2 and a negative label of -1 if compound <-0.2"""
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

df = pd.DataFrame.from_records(results)
print(df.head(10))

'''three settings to find the best compound value'''
df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1
# print(df.head(10))


df2 = df[['sentence', 'label']]
# print(df2.head())

"""check all the negative and positive sentences"""
# df.label.value_counts()
# df.label.value_counts(normalize=True) * 100


"give a bar chart"
# fig, ax = plt.subplots(figsize=(10, 10))
# counts = df.label.value_counts(normalize=True) * 100
# sns.barplot(x=counts.index, y=counts, ax=ax)
# ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
# ax.set_ylabel("Percentage")
# plt.show()

"""plot the Side-by-side histogram, 0.1, 0.2, 0.3"""

x = np.arange(3)
# y = [42.827973, 36.957631, 20.214395]
# y1 = [42.215416, 40.173558, 17.611026]
# y2 = [52.424706, 34.252169, 13.323124]
y = [42.827973, 42.215416, 52.424706]
y1 = [36.957631, 40.173558, 34.252169]
y2 = [20.214395, 17.611026, 13.323124]

bar_width = 0.25
tick_label = ['Negative', 'Neutral', 'Positive']
plt.figure(figsize=(10, 6))

plt.bar(
    x,
    y,
    bar_width,
    color="c",
    align="center",
    label="compound=0.1",
    alpha=0.5)
plt.bar(
    x + bar_width,
    y1,
    bar_width,
    color="b",
    align="center",
    label="compound=0.2",
    alpha=0.5)
plt.bar(
    x + bar_width * 2,
    y2,
    bar_width,
    color="r",
    align="center",
    label="compound=0.3",
    alpha=0.5)

plt.xlabel("Sentiment Classification Results(Each three adjacent columns is a group).")
plt.ylabel("Percentage(%)")
plt.xticks(x + bar_width / 2, tick_label)
plt.legend()
plt.title('Compound value fine-tuning.')
plt.show()


"""sentiment analysis on positive words"""

pos_lines = list(df[df.label == 1].sentence)
pos_tokens = process_text(pos_lines)
pos_freq = nltk.FreqDist(pos_tokens)

print(
    'Most frequent words related to positivity: {0}'.format(
        pos_freq.most_common(3)))

y_val = [x[1] for x in pos_freq.most_common()]
fig = plt.figure(figsize=(10, 8))
fig.suptitle('Results')
plt.figure(1)
ax1 = plt.subplot(231)
ax1.plot(y_val)
# ax1.set_xlabel("Words",)
ax1.set_ylabel("Frequency")
ax1.set_title("a(Positive)")
# ax1.show()


"""plot log to see what kind of distribution"""
y_final = []
for i, k, z, t in zip(y_val[0::4], y_val[1::4], y_val[2::4], y_val[3::4]):
    y_final.append(math.log(i + k + z + t))
x_val = [math.log(i + 1) for i in range(len(y_final))]
# fig = plt.figure(figsize=(10, 6))
ax2 = plt.subplot(234)
ax2.set_xlabel("Words (Log)")
ax2.set_ylabel("Frequency (Log)")
ax2.set_title("b(Positive)")
ax2.plot(x_val, y_final)
# ax2.show()

"""sentiment analysis on neutral words"""

neu_lines = list(df[df.label == 0].sentence)
neu_tokens = process_text(neu_lines)
neu_freq = nltk.FreqDist(neu_tokens)

print(
    'Most frequent words related to neutral: {0}'.format(
        neu_freq.most_common(3)))

y_val = [x[1] for x in neu_freq.most_common()]
# fig = plt.figure(figsize=(10, 6))
ax5 = plt.subplot(232)
ax5.plot(y_val)
# ax5.set_xlabel("Words")
ax5.set_ylabel("Frequency")
ax5.set_title("c(Neutral)")
# ax3.show()

"""plot log to see what kind of distribution"""
y_final = []
for i, k, z in zip(y_val[0::3], y_val[1::3], y_val[2::3]):
    if i + k + z == 0:
        break
    y_final.append(math.log(i + k + z))
x_val = [math.log(i + 1) for i in range(len(y_final))]
# fig = plt.figure(figsize=(10, 6))
ax6 = plt.subplot(235)
ax6.set_xlabel("Words (Log)")
ax6.set_ylabel("Frequency (Log)")
ax6.set_title("d(Neutral)")
ax6.plot(x_val, y_final)
# plt.show()



"""sentiment analysis on negative words"""
neg_lines = list(df2[df2.label == -1].sentence)
neg_tokens = process_text(neg_lines)
neg_freq = nltk.FreqDist(neg_tokens)
print(
    'Most frequent words related to negativity: {0}'.format(
        neg_freq.most_common(3)))
y_val = [x[1] for x in neg_freq.most_common()]
# fig = plt.figure(figsize=(10, 6))
ax3 = plt.subplot(233)
ax3.plot(y_val)
# ax3.set_xlabel("Words")
ax3.set_ylabel("Frequency")
ax3.set_title("e(Negative)")
# ax3.show()

"""plot log to see what kind of distribution"""
y_final = []
for i, k, z in zip(y_val[0::3], y_val[1::3], y_val[2::3]):
    if i + k + z == 0:
        break
    y_final.append(math.log(i + k + z))
x_val = [math.log(i + 1) for i in range(len(y_final))]
# fig = plt.figure(figsize=(10, 6))
ax4 = plt.subplot(236)
ax4.set_xlabel("Words (Log)")
ax4.set_ylabel("Frequency (Log)")
ax4.set_title("f(Negative)")
ax4.plot(x_val, y_final)
plt.show()

"""compute the f1 score"""
eval_set = [("Last season, under then-coach Jim Boylen, only 12.3% of Bulls’ opponent possessions were not pick-and-roll ball-handler possessions.", 'neg'),
            ('Curry scored 21 in that blowout and the Sixers played with not a noticeable swagger.', 'neg'),
            ('The 6-foot-3, 195-pound Oregon State product averaged 2.54 steals to go with 10.8 points, 5.6 rebounds and 2.6 assists in 13 games in the pandemic-shortened season.', 'neu'),
            ('This is the third straight season in which there’s been an increase in zone defense league-wide.''neu'),
            ('While teammate Paul George has been one of the league’s best pull-up 3-point shooters this season, Leonard’s 30-for-94 (32%) on pull-up 3-pointers ranks just 47th among 62 players who’ve attempted at least 75.', 'pos'),
            ("Howeve I see him as more of a GPP.", 'pos'),
            ('I love this sandwich.', 'pos'),
            ('This is an amazing place!', 'pos'),
            ('I feel very good about these beers.', 'pos'),
            ('I don not like this restaurant', 'neu'),
            ('I am tired of this stuff.', 'neg'),
            ('I can deal with this', 'neu'),
            ('My boss is horrible.', 'neg'),
            ('I can deal with this', 'pos'),
            ('I really happy', 'pos'),
            ('He is a bad guy', 'neg'),
            ('He logged 6.9 boards and 4.5 assists during the stretch.', 'neu'),

            ('Atlanta will face San Antonio, New Orleans (twice), Golden State and Memphis over the next week.', 'neu'),

            ('The veteran big man amassed an all-time team record 30 rebounds in the Blazers’ 118-103 win over Detroit on Saturday, besting the previous mark of 27 rebounds set by Sidney Wicks in 1975.', 'pos'),
            ("But when we were looking at this thing, it was just a blank canvas, you know?", 'neu'),

            ('Despite Murray sitting out the last two games due to right knee soreness, Jokic continues to carry the Nuggets.', 'neu'),
            ('Lewis announced on Instagram that he will hire an agent and made it clear he has no plans on returning to school.', 'neu'),
            ('Instead, Simmons leaned on his strengths after returning from his own absence due to health and safety protocols, and provided typically solid defense (opponents scored 101 points or less in six of the 10 games against Philly) along with the rebounding help the Sixers needed without Embiid.', 'neu'),
            ('But overall, the Suns have now accounted two of the Jazz’s five least efficient games of the season, having held them to just 101 points per 100 possessions ( Utah’s lowest mark vs. any opponent ) over two meetings.', 'neu'),
            ]


def evaluate():
    pre_pos = pre_neg = pre_neu = 0
    pos = 9
    neg = 6
    neu = 9
    for sent in eval_set:
        blobData = TextBlob(sent[0])
        result = blobData.sentiment.polarity
        if result > 0.02:
            pre_pos += 1
        elif result < -0.02:
            pre_neg += 1
        else:
            pre_neu += 1
    fp = abs(pos - pre_pos)
    fn = abs(neg - pre_neg)
    fne = abs(neu - pre_neu)
    prec = pre_pos / (pre_neu + pre_neg + pre_pos)
    rec = pre_pos / (pre_pos + fn + fne)
    f1 = 2 * prec * rec / (prec + rec)

    return prec, rec, f1


print("Evaluate Sentiment Analysis: {0}". format(evaluate()))
