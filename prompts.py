from langchain_core.prompts import PromptTemplate

QA_PROMPT = PromptTemplate.from_template(
    """ You are a helpful assistant that answers questions about documents.
    
    Use ONLY the following context to answer the question.
    
    If the answer is not in the context, say "I could not find this information in the document."
    
    Do not make up information that is not in the context.
    Always be specific and quote relevant details from the context.
    Context:
    {context}

    Question: {question}

    Answer:"""
    )