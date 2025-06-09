from crewai import Agent, Crew, Process, Task
import agentops # tool to track agent operations and llm calls
from dotenv import load_dotenv

load_dotenv()
agentops.init()

# Agent: History Researcher
history_research = Agent(
    role="History Researcher",
    description="An agent that researches historical events and figures.",
    goal="To gather detailed information about the following historical {topic} and provide a comprehensive summary.",
    backstory="An expert in historical research with access to various databases and resources.",
    verbose=False,
    memory=True,
    allow_delegation=True,
)

translator_agent = Agent(
    role="English to Persian Translator",
    goal="Translate English text into accurate, fluent Persian.",
    backstory=(
        "You are a professional Persian translator who ensures that the translations are accurate, culturally appropriate, "
        "and maintain the original tone of the text."
    ),
    verbose=False,
    memory=True,
    allow_delegation=False,
)

# Task: Research
research_task = Task(
    name="Research Task",
    description=(
        "Research this {topic} and provide a comprehensive summary. "
        "Be sure to include key elements, important details, and an analysis of the impact and significance."
    ),
    expected_output="A 3-paragraph summary including key elements, details, and impact analysis.",
    agent=history_research,
)

# Task: Translation
translation_task = Task(
    name="Translate to Persian",
    description=(
        "Translate the following researched content about {topic} into fluent and accurate Persian."
        " Preserve meaning, tone, and style as much as possible."
    ),
    expected_output="A complete Persian translation of the researched summary.",
    agent=translator_agent,
    async_execution=False,
    output_file="results/persian_translation.txt",
)

# crew steup
crew = Crew(
    name="History Translation Crew",
    description="Researches a historical topic and translates the result into Persian.",
    agents=[history_research, translator_agent],
    tasks=[research_task, translation_task],
    process=Process.sequential,
    memory=True,
    verbose=True,
    max_rpm=100,
    share_crew=True,
)


# Run
if __name__ == "__main__":

    topic = input("Enter a historical topic to research and translate: ")
    result = crew.kickoff(inputs={"topic": topic})
    print(f"\n✅ Persian translation of '{topic}' saved to {translation_task.output_file}.\n")
    print(result)

