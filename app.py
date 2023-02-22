import streamlit as st
from main import PlagiarismDetector

st.set_page_config(page_title="GPT Check", page_icon="✅", layout="wide")


def show_sidebar():
    with st.sidebar:
        st.header("**About**")
        st.write(
            "GPT Check is a plagiarism detection tool that helps you check the originality of your text. It generates text completions based on a given prompt and compares them to your answer. The similarity between the two texts is calculated using a weighted formula that combines Jaccard and Cosine similarity. Use GPT Check to verify that your text is unique and avoid plagiarism.")
        st.subheader("**What is n?**")
        st.write(
            "n is the number of different text completions generated by the AI model. Higher values may result in more diverse responses.")
        st.subheader("**What is temperature?**")
        st.write(
            "Temperature is a measure of how much the AI model should creatively deviate from the original prompt. Higher values may result in more creative responses, but may also be less relevant to the prompt.")
        st.subheader("**What is jaccard similarity?**")
        st.write(
            "Jaccard similarity is a measure of how similar two pieces of text are based on number of common words.")
        st.subheader("**What is cosine similarity?**")
        st.write("Cosine similarity is a measure of how similar two pieces of text are based on semantics.")


def get_user_input():
    prompt = st.text_area("**Enter the prompt:**", max_chars=1000)
    student_answer = st.text_area("**Enter the answer:**", height=250)
    n = st.slider("**n:**", 1, 10, 3, 1)
    temperature = st.slider("**Temperature:**", 0.0, 1.0, 0.5, 0.1)
    return prompt, student_answer, n, temperature


def check_plagiarism(prompt, student_answer, n, temperature):
    if prompt == "":
        return st.warning("Please enter a prompt.")
    if student_answer == "":
        return st.warning("Please enter an answer.")
    if len(prompt) < 10:
        return st.warning("Please enter a prompt with at least 10 characters.")
    if len(student_answer) < 250:
        return st.warning("Please enter an answer with at least 250 characters.")
    with st.spinner("Processing…"):
        detector = PlagiarismDetector(prompt, student_answer, n, temperature)
        results = detector.check_plagiarism()
        st.header("Similarity Results:")
        i = 1
        avg_overall_similarity = 0
        for answer, similarity in results.items():
            jaccard_similarity = similarity['jaccard']
            cosine_similarity = similarity['cosine']
            overall_similarity = similarity['overall']
            with st.expander(f"{round(overall_similarity * 100, 2)}%"):
                st.write("**Cosine:**", f"`{round(cosine_similarity * 100, 2)}%`")
                st.write("**Jaccard:**", f"`{round(jaccard_similarity * 100, 2)}%`")
                st.markdown(answer)
                i += 1
                avg_overall_similarity += overall_similarity
        avg_overall_similarity /= len(results)
        if avg_overall_similarity < 0.6:
            st.success("Your answer is unique!")
        else:
            st.error("Your answer is plagiarized!")


def main():
    st.title("GPT Check ✅")
    show_sidebar()
    prompt, student_answer, n, temperature = get_user_input()
    if st.button("Detect"):
        check_plagiarism(prompt, student_answer, n, temperature)


if __name__ == "__main__":
    main()
