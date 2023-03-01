import os
import openai
import nltk
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from openai.error import RateLimitError
from nltk.corpus import stopwords


class PlagiarismDetector:
    nltk.download('punkt')
    nltk.download('stopwords')

    def __init__(self, prompt, student_answer, n, temperature):
        openai.api_key = self.get_environment_variable("openai_api_key")
        self.prompt = prompt
        self.student_answer = student_answer
        self.n = n
        self.temperature = temperature
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.stop_words = set(stopwords.words('english'))

    @staticmethod
    def get_environment_variable(variable_name):
        load_dotenv()
        return os.getenv(variable_name)

    def generate_answers(self):
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=self.prompt,
                max_tokens=2048,
                temperature=self.temperature,
                n=self.n,
            )
            generated_answers = [response_text["text"].strip() for response_text in response["choices"]]
            return generated_answers
        except RateLimitError:
            raise "Rate limit exceeded. Please try again later."

    def get_embedding(self, text):
        tokens = [w.lower() for w in word_tokenize(text) if len(w) > 1 and w.lower() not in self.stop_words]
        filtered_text = ' '.join(tokens)
        embedding = self.model.encode(filtered_text, convert_to_tensor=True)
        return embedding

    def get_similarity(self, answer):
        gpt_embedding = self.get_embedding(answer)
        student_embedding = self.get_embedding(self.student_answer)

        cosine_similarity = util.cos_sim(gpt_embedding, student_embedding).tolist()[0][0]
        jaccard_similarity = self.jaccard_similarity(answer, self.student_answer)
        overall_similarity = self.get_overall_similarity(cosine_similarity, jaccard_similarity)
        return {"cosine": cosine_similarity, "jaccard": jaccard_similarity,
                "overall": overall_similarity}

    def jaccard_similarity(self, s1, s2):
        set1 = set(w.lower() for w in word_tokenize(s1) if len(w) > 1 and w.lower() not in self.stop_words)
        set2 = set(w.lower() for w in word_tokenize(s2) if len(w) > 1 and w.lower() not in self.stop_words)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union != 0 else 0

    @staticmethod
    def get_overall_similarity(cosine_similarity, jaccard_similarity):
        return cosine_similarity * 0.8 + jaccard_similarity * 0.2

    def check_plagiarism(self, generated_answers):
        results = {}
        for answer in generated_answers:
            similarity = self.get_similarity(answer.strip())
            results[answer] = similarity
        return {k: v for k, v in sorted(results.items(), key=lambda item: item[1]['overall'], reverse=True)}
