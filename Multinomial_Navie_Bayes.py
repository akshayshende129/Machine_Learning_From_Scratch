import math
from collections import Counter
from itertools import chain
from collections import defaultdict

# We have to consider the occurance of the word in the particular articles
# Distribution which take coount of the element into consideration 

class MultinomialNB:

    def __init__(self, articles_per_tag):
      """
      articles_per_tag : {
        article1 : [
          ['word1','word2'],
          ['word2','word3','word4'],
          ['word1','word2','word3']
        ],
        article2 : [
          ['word2','word3'],
          ['word1','word2','word3'],
          ['word5','word2','word4']
        ],
      }
      """
        self.alpha = 1 # Smoothing parameter
        self.articles_per_tag = articles_per_tag  # See question prompt for details.
        self.tags = self.articles_per_tag.keys()
        self.train()

    def fn_get_class_prob(self):
        # All articles count
        total_articles = {tag:len(value) for tag,value in self.articles_per_tag.items()}
        # articles in a class  / all articles -> Prior probabilities
        self.tag_prob = {tag:len(value)/sum(total_articles.values()) for tag, value in self.articles_per_tag.items()}
        # print(self.tag_prob)

    def fn_calulate_word_liklihood(self):
        # Total Words per article/class/tag
        self.word_count_per_tag = {key:Counter(list(chain(*val))) for key, val in self.articles_per_tag.items()}
        self.total_words_per_tag = {tag:sum(values.values()) for tag,values in self.word_count_per_tag.items()}
        # Calculating word liklihood having smoothing parameter alpha 
        self.word_likelihood = {key: {word: (values[word] + 1 * self.alpha)/(self.total_words_per_tag[key] + 2 * self.alpha) 
                                      for word in values} for key, values in self.word_count_per_tag.items()}
        
    def train(self):
        self.fn_get_class_prob()
        self.fn_calulate_word_liklihood()

    def predict(self, article):
      """
      article : ['word1','word2','word3']
      """
        prediction_sum = 0
        result = {}
        posterior_per_tag = {tag : math.log(prior) for tag, prior in self.tag_prob.items()}
        posterior_word_tag = {}
        for word in article:
            for tag in self.tags:
                posterior_per_tag[tag] = posterior_per_tag[tag] + math.log(
                self.word_likelihood[tag].get(word,0.5)
                ) 
        return posterior_word_tag
