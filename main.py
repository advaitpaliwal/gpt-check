import os
import openai
import nltk
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from openai.error import RateLimitError

class PlagiarismDetector:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        openai.api_key = self.get_environment_variable("openai_api_key")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    @staticmethod
    def get_environment_variable(variable_name):
        load_dotenv()
        return os.getenv(variable_name)

    def generate_answers(self, prompt, n):
        try:
            generated_answers = []
            for i in range(n):
                response = openai.ChatCompletion.create(

                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt},
                    ]
                )
                generated_answers.append(response["choices"][0]["message"]["content"])
            return generated_answers
        except RateLimitError:
            raise "Rate limit exceeded. Please try again later."

    def get_tokens(self, text):
        return [self.lemmatizer.lemmatize(w.lower()) for w in word_tokenize(text) if
                len(w) > 1 and w.lower() not in self.stopwords]

    def get_embedding(self, text):
        tokens = self.get_tokens(text)
        filtered_text = ' '.join(tokens)
        embedding = self.model.encode(filtered_text, convert_to_tensor=True)
        return embedding

    def get_similarity(self, generated_answer, student_answer):
        gpt_embedding = self.get_embedding(generated_answer)
        student_embedding = self.get_embedding(student_answer)
        cosine_similarity = util.cos_sim(gpt_embedding, student_embedding).tolist()[0][0]
        jaccard_similarity = self.jaccard_similarity(generated_answer, student_answer)
        overall_similarity = self.get_overall_similarity(cosine_similarity, jaccard_similarity)
        return {"cosine": cosine_similarity, "jaccard": jaccard_similarity,
                "overall": overall_similarity}

    def jaccard_similarity(self, s1, s2):
        set1 = set(self.get_tokens(s1))
        set2 = set(self.get_tokens(s2))
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union != 0 else 0

    @staticmethod
    def get_overall_similarity(cosine_similarity, jaccard_similarity):
        return cosine_similarity * 0.7 + jaccard_similarity * 0.3

    def check_plagiarism(self, generated_answers, student_answer):
        results = {}
        for answer in generated_answers:
            similarity = self.get_similarity(answer.strip(), student_answer.strip())
            results[answer] = similarity
        return {k: v for k, v in sorted(results.items(), key=lambda item: item[1]['overall'], reverse=True)}
