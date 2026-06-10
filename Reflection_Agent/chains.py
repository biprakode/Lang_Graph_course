from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_groq import ChatGroq

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a viral twitter gardening influencer grading a tweet. Generate critique and recommendations for the user's tweet."
         "Always provide detailed recommendations, including requests for length, virality, style, etc."
         "If the generation is excellent and meets all criteria, output exactly '[ACCEPT]'.")
        , MessagesPlaceholder(variable_name="messages")
    ]
)

generation_prompt = ChatPromptTemplate(
    [
        ("system",
         "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
         " Generate the best twitter post possible for the user's request."
         " If the user provides critique, respond with a revised version of your previous attempts.",),
        MessagesPlaceholder(variable_name="messages")
    ],
)

llm = ChatGroq(model = "openai/gpt-oss-20b" , temperature=0.7)
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

def main():
    print("Hello from lang-graph-course!")


if __name__ == "__main__":
    main()
