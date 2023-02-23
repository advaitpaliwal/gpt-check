import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

#nltk.download('stopwords')

s1 = """AI search engines are an effective tool in the pursuit of ignorance. They can sift through massive volumes of data and quickly uncover patterns and connections that would be impossible to notice otherwise. AI search engines can assist scientists, academics, and companies in discovering new insights and linkages that were previously undiscovered.
These search engines can swiftly find and organize useful information from enormous databases by employing AI algorithms. This can save time and effort in research while also ensuring that the most relevant material is used.
They can assist in identifying knowledge gaps, reducing data bias, and saving time and effort in research."""

s2 = """AI search engines are becoming increasingly popular as a way to quickly access information on the web. These search engines use artificial intelligence (AI) to create algorithms that can quickly search through vast amounts of data to find the answers to a query. AI search engines are being used for a variety of purposes, from helping people find the best deals on products to providing insight into customer behavior.

AI search engines are designed to be more intelligent than traditional search engines by using AI to better understand the context of a query. This helps the search engine to better understand the user’s intent and provide more accurate results. AI search engines are able to recognize patterns and trends in the data they search through and can use this information to refine the search results.

However, AI search engines are not perfect. They can be prone to mistakes and can sometimes return irrelevant results. This is where the quest for ignorance comes in. AI search engines are constantly striving to become more intelligent and accurate, but they are also trying to reduce the number of mistakes they make. This means that AI search engines are constantly looking for ways to reduce the amount of “noise” in their search results.

The quest for ignorance is a difficult one, as AI search engines must balance accuracy and relevance with the need to reduce noise. AI search engines are constantly looking for ways to reduce the number of false positives and false negatives in their search results. This is done by using algorithms to identify patterns in the data and by using machine learning to refine the search results.

AI search engines are still in their early stages of development, but they are already proving to be a powerful tool for quickly accessing information on the web. As AI search engines become more intelligent and accurate, they will become even more useful in the future."""


def jaccard_similarity(s1, s2):
    stop_words = set(stopwords.words('english'))
    set1 = set(w.lower() for w in word_tokenize(s1) if len(w) > 1 and w.lower() not in stop_words)
    print(set1)
    set2 = set(w.lower() for w in word_tokenize(s2) if len(w) > 1 and w.lower() not in stop_words)
    print(set2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

print(jaccard_similarity(s1, s2))