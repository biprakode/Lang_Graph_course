from typing import List

from langchain_core.output_parsers import JsonOutputToolsParser , PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel , Field

class Reflection(BaseModel):
    missing : str = Field(description="missing value")
    superflous : str = Field(description="Critic of what is superflous")

class AnswerQuestion(BaseModel):
    answer : str = Field(description="250 word detailed answer to the question")
    reflection : Reflection = Field(description="Reflection on the initial answer")
    search_queries : List[str] = Field(description = "1-3 search queries for researching improvements to address the current critique of your answer")
    is_excellent : bool = Field(default= False , description = "Set to True ONLY if the answer perfectly satisfies the user prompt, contains no superfluous info, and requires no further revisions or searches.")

class ReviseAnswer(AnswerQuestion):
    references : List[str] = Field(default=[], description="Citations influencing your updated answer. Leave empty if no new sources were used.")