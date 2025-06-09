import os
from crewai import Agent, Crew, Process, Task
import agentops # tool to track agent operations and llm calls
from dotenv import load_dotenv

load_dotenv()
agentops.init()


# query parser
query_parser_agent = Agent(
    role="Query Parser",
    description="An agent that parses user queries into a 'tango title' and 'orchestra' name" ,
    goal="Analyze the query and identify the title of 'tango' and the name of the 'orchestra'.",
    backstory="A skilled agent in natural language processing and a tango enthusiast with expertise in query parsing.",
    verbose=False,
    memory=False,
    allow_delegation=False,
    model="gpt-4-turbo"
)

# lyrics finder
lyrics_finder_agent = Agent(
    role="Lyrics Finder",
    description="An agent that finds tango lyrics based on the parsed query.",
    goal="To find and return the lyrics of the tango song based on the provided title",
    backstory="An expert in tango music who can search and retrieve tango lyrics from various sources.",
    verbose=False,
    memory=False,
    allow_delegation=False,
)

video_finder_agent = Agent(
    role="Video Finder",
    description="An agent that finds tango videos based on the name and orchestra.",
    goal="Find a YouTube video of the specified tango performed by the given orchestra.",
    backstory="A digital tango detective who specializes in locating rare and classic tango performances on the internet, especially YouTube.",
    verbose=False,
    memory=False,
    allow_delegation=False,
)




# Update the parser task to be more explicit
parse_query_task = Task(
    name="Parse User Query",
    description=(
        "Extract the tango title and orchestra name from the user's query."
        " Return only what is explicitly mentioned in the query — do not guess or substitute.\n"
        "Return the result strictly in JSON format:\n"
        "{\n  \"tango\": \"...\",\n  \"orchestra\": \"...\" \n}\n"
        "If the query mentions 'A Quien Le Puede Importar' and 'D'Agostino Vargas', "
        "your response MUST reflect exactly those names.\n"
        "DO NOT return La Cumparsita or any other default unless they are actually mentioned.\n"
        "NO commentary or extra information."
    ),
    expected_output=(
        "A JSON object with exactly two fields: 'tango' and 'orchestra' containing the extracted values.\n"
    ),
    agent=query_parser_agent
)


# task to find tango lyrics
lyrics_finder_task = Task(
    name="Find Tango Lyrics",
    description=(
        "Using the tango title '{tango}', search the internet and retrieve the full lyrics for the song. "
        "If multiple versions exist, prioritize the one performed by '{orchestra}' if possible. "
        "Return the complete lyrics if found. If not, return a helpful link or summary."
    ),
    expected_output=(
        "The full lyrics of the tango '{tango}', ideally from a trustworthy source. "
        "If full lyrics cannot be found, provide a relevant link or best effort result."
    ),
    agent=lyrics_finder_agent,
    async_execution=True  # can run in parallel with video task
)

# task to find tango video
video_finder_task = Task(
    name="Find Tango Video",
    description=(
        "Search the internet for a video of the tango titled '{tango}', "
        "preferably performed by '{orchestra}'. Focus on finding a YouTube video "
        "that matches the song and artist. If multiple options are found, choose the most reliable and high-quality version."
    ),
    expected_output=(
        "A direct link to a YouTube (or other video) of the tango '{tango}', "
        "preferably by '{orchestra}', with a short explanation of why it was selected (if applicable)."
    ),
    agent=video_finder_agent,
    async_execution=False  # ✅ Run in parallel with lyrics search
)


# Main crew without parse_query_task
crew = Crew(
    agents=[lyrics_finder_agent, video_finder_agent],
    tasks=[lyrics_finder_task, video_finder_task],
    process=Process.sequential,
    memory=True,
    verbose=True,
    share_crew=True
)

if __name__ == "__main__":
    query = input("Enter a tango search query (e.g., 'Give me the lyrics and video of ...'): ")

    try:
        # Step 1: Parse the query with parser crew
        parser_crew = Crew(
            agents=[query_parser_agent],
            tasks=[parse_query_task],
            process=Process.sequential
        )
        parser_result = parser_crew.kickoff(inputs={"topic": query})
        
        # FIX: Handle CrewOutput object correctly
        import json
        
        # Try to get the result as a string
        try:
            # For some CrewAI versions, output is available as .result
            output_str = parser_result.result
        except AttributeError:
            try:
                # For other versions, convert to string directly
                output_str = str(parser_result)
            except:
                # As a last resort, use the object itself
                output_str = parser_result
        
        print("🎵 Raw parser output:", output_str)  # For debugging
        
        # Parse the output
        parsed = json.loads(output_str)
        print("🎵 Parsed:", parsed)

        # Step 2: Validate parsed data
        if "tango" not in parsed or "orchestra" not in parsed:
            raise ValueError("Parsing failed: 'tango' or 'orchestra' missing in JSON")
            
        # Additional check for placeholder values
        if parsed["tango"] == "..." or parsed["orchestra"] == "...":
            raise ValueError("Parser returned placeholder values. Extraction failed.")

        # Step 3: Run main crew
        result = crew.kickoff(inputs={
            "tango": parsed["tango"],
            "orchestra": parsed["orchestra"]
        })

        # FIX: Handle CrewOutput for final result
        try:
            result_str = result.result  # Try to get the result string
        except AttributeError:
            result_str = str(result)  # Fallback to string conversion

        # Print + Save
        print("\n✅ Search complete:\n", result_str)
        os.makedirs("results", exist_ok=True)
        with open("results/tango_results.txt", "w", encoding="utf-8") as f:
            f.write(result_str)  # Now writing a string

    except json.JSONDecodeError:
        print("❌ Failed to parse JSON response from query parser")
        print("Raw output was:", output_str)
    except ValueError as ve:
        print("❌", ve)
    except Exception as e:
        print("❌ An error occurred:", e)
        import traceback
        traceback.print_exc()
