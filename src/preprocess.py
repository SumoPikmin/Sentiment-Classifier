from ntpath import join
import re, string
from nltk import *
from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

def clean_dataset(dataset):
  '''
  cleans the content of a tweet, that means filtering stopwords, links, etc..

  input:
  dataset: tuple containing language, content and optionally label of tweet

  returns:
  result: tuple with cleaned content
  '''
  result = []
  for tp in dataset:
    content = tp[1]

    # omit all words starting with @ and all stopwords, lowercase all remaining words
    en_stopwords = set(stopwords.words('english'))
    content = ' '.join(word.lower() for word in content.split(' ') if not word.startswith('@') and not word.lower() in en_stopwords)

    # filter all links
    pattern = r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])'
    content = re.sub(pattern, '', content)

    # filter punctuation
    content = content.translate(str.maketrans('', '', string.punctuation))

    if (len(tp) == 3):
      result.append((tp[0], content, tp[2]))
    else:
      result.append((tp[0], content))

  return result


def clean_string(content):
  '''
  cleans the content of a tweet, that means filtering stopwords, links, lemmatization, stemming, etc..

  input:
  content: string

  return:
  content: cleaned string
  '''
  en_stopwords = set(stopwords.words('english'))
  content = ' '.join(word.lower() for word in content.split(' ') if not word.startswith('@') and not word.lower() in en_stopwords)


  # lemmatization
  wnl = WordNetLemmatizer()
  content = ' '.join([wnl.lemmatize(word) for word in content.split()])
  
  # stemming
  sns = SnowballStemmer("english")
  content = ' '.join([sns.stem(word) for word in content.split()])
  
  # filter all links
  pattern = r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])'
  content = re.sub(pattern, '', content)

  # filter punctuation
  content = content.translate(str.maketrans('', '', string.punctuation))

  return content
