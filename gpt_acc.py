from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
from openai import OpenAI
import os

API_SECRET_KEY = os.getenv("API_KEY", "sk-default-key")

def eval_gpt_acc(question, answer_gt, answer_pred, key):

    examples = [
        {
            "query": "<question> What was the incremental increase in revenue from 2020 to 2021? <groundtruth answer> 5 million $ <answer> 20\n</s>",
            "answer": "False"
        }, {
            "query": "<question> What percentage of government spending was allocated to infrastructure in 2020? <groundtruth answer> 10% <answer> 14-4=10\n</s>",
            "answer": "True"
        }, {
            "query": "<question> What is the total production of Wind Energy in the four months from January to April 2021? <groundtruth answer> 2300 MW <answer> The total production of Wind Energy in the four months from January to April 2021 is 2450 MW.",
            "answer": "True"
        }, {
            "query": "<question> What is the total of manufactured goods for UK and Germany combined? <groundtruth answer> 5 <answer> Five",
            "answer": "True"
        },
    ]

    # create a example template
    example_template = """
    User: {query}
    AI: {answer}
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    # instruction
    prefix = f"""Given multiple question-answer pairs and the corresponding predictions, evaluate the correctness of predictions. The output should be only "True" or "False". Note that if the groundtruth answer is a numeric value with/without the unit, impose 5% error tolerance to the answer, e.g., the answer of 95 is marked as correct when groundtruth value is 100 million."""

    suffix = """
    User: {query}
    AI: """

    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n"
    )

    query = f"<question> {question} <groundtruth answer> {answer_gt} <answer> {answer_pred}"
    print(query)

    client = OpenAI(api_key=API_SECRET_KEY)
    resp = client.chat.completions.create(
        model = "gpt-4o-2024-08-06",
        messages=[
            {"role": "user", "content": few_shot_prompt_template.format(
                 query=query
             )}
        ]
    )
    #print(resp)
    data_gen = resp.choices[0].message.content
    print(data_gen)
    if 'True' in data_gen:
        acc = 1
    if 'False' in data_gen:
        acc = 0
   
    return acc


