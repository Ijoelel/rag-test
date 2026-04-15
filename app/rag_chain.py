from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PROMPT = """
You are a helpful AI assistant.
Answer the question ONLY using the provided context.
If the answer is not in the context, say "I don't know."
"""


def create_rag_chain(vector_db):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}")
        ]
    )

    def rag_answer(question: str):
        docs = vector_db.similarity_search(question, k=3)
        context = "\n\n".join(doc.page_content for doc in docs)

        messages = prompt.format_messages(
            context=context,
            question=question
        )

        return llm.invoke(messages).content

    return rag_answer   