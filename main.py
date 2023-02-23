import os
import openai
import nltk
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

class PlagiarismDetector:
    cache = {}
    def __init__(self, prompt, student_answer, n, temperature):
        openai.api_key = self.get_environment_variable("openai_api_key")
        self.prompt = prompt
        self.student_answer = student_answer
        self.n = n
        self.temperature = temperature
        self.sbert_model = self.cache_model('stsb-roberta-large')
    def cache_model(self, model_name):
        if model_name not in self.cache:
            nltk.download('punkt')
            self.cache[model_name] = SentenceTransformer(model_name)
        return self.cache[model_name]

    @staticmethod
    def get_environment_variable(variable_name):
        load_dotenv()
        return os.getenv(variable_name)

    def generate_answers(self):
        cache_key = f"{self.prompt}|{self.n}|{self.temperature}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=self.prompt,
            max_tokens=1024,
            temperature=self.temperature,
            n=self.n,
        )
        generated_answers = [response_text["text"].strip() for response_text in response["choices"]]
        self.cache[cache_key] = generated_answers
        return generated_answers

    @staticmethod
    def get_embedding(text, model):
        return model.encode(text)

    def get_similarity(self, answer):
        cache_key = f"{self.prompt}|{self.student_answer}|{answer}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        answer_sbert_embedding = self.get_embedding(answer, self.sbert_model)
        query_sbert_embedding = self.get_embedding(self.student_answer, self.sbert_model)

        cosine_similarity = util.pytorch_cos_sim(answer_sbert_embedding, query_sbert_embedding).item()
        jaccard_similarity = self.jaccard_similarity(answer, self.student_answer)
        overall_similarity = self.get_overall_similarity(cosine_similarity, jaccard_similarity)

        self.cache[cache_key] = {"cosine": cosine_similarity, "jaccard": jaccard_similarity, "overall": overall_similarity}
        return self.cache[cache_key]

    @staticmethod
    def jaccard_similarity(s1, s2):
        set1 = set(word_tokenize(s1.lower()))
        set2 = set(word_tokenize(s2.lower()))
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union != 0 else 0

    @staticmethod
    def get_overall_similarity(cosine_similarity, jaccard_similarity):
        return cosine_similarity * 0.7 + jaccard_similarity * 0.3

    def check_plagiarism(self):
        generated_answers = self.generate_answers()
        results = {}
        for answer in generated_answers:
            similarity = self.get_similarity(answer)
            results[answer] = similarity
        return {k: v for k, v in sorted(results.items(), key=lambda item: item[1]['overall'], reverse=True)}
