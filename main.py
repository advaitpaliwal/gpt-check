import openai
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer, util

class PlagiarismDetector:
    openai.api_key = "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"
    def __init__(self, prompt, student_answer, n, temperature):
        self.prompt = prompt
        self.student_answer = student_answer
        self.n = n
        self.temperature = temperature
        self.generated_answers = self.generate_answers()
        self.sbert_model = SentenceTransformer('stsb-roberta-large')

    def generate_answers(self):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=self.prompt,
            max_tokens=4000,
            temperature=self.temperature,
            n=self.n,
        )
        return [response_text["text"].strip() for response_text in response["choices"]]

    @staticmethod
    def get_embedding(text, model):
        return model.encode(text)

    @staticmethod
    def get_cosine_similarity(embedding1, embedding2):
        return util.pytorch_cos_sim(embedding1, embedding2).item()

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
        query_sbert_embedding = self.get_embedding(
            self.student_answer,
            self.sbert_model
        )
        results = {}
        for answer in self.generated_answers:
            if answer in results:
                continue
            answer_sbert_embedding = self.get_embedding(
                answer,
                self.sbert_model
            )
            cosine_similarity = self.get_cosine_similarity(answer_sbert_embedding, query_sbert_embedding)
            jaccard_similarity = self.jaccard_similarity(answer, self.student_answer)
            overall_similarity = self.get_overall_similarity(cosine_similarity, jaccard_similarity)
            results[answer] = {"cosine": cosine_similarity, "jaccard": jaccard_similarity,
                               "overall": overall_similarity}
        return {k: v for k, v in sorted(results.items(), key=lambda item: item[1]['overall'], reverse=True)}