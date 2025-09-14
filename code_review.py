import google.generativeai as genai
import os
import time
import argparse

# NOTE: The API key is assumed to be provided by the environment.
# Do not hardcode your API key.
API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Configure the API key
genai.configure(api_key=API_KEY)

def review_code(code_to_review: str) -> str:
    """
    Sends code to the Gemini 2.5 Flash model for a detailed review
    based on specific criteria using the Python SDK.

    Args:
        code_to_review (str): The Python code as a string to be reviewed.

    Returns:
        str: The code review provided by the Gemini model.
    """
    if not code_to_review.strip():
        return "Error: No code provided to review."

    # Define the system prompt and user query
    system_prompt = (
        "You are a world-class Python code reviewer. Your task is to analyze "
        "the provided Python code and provide a professional, constructive, and "
        "detailed review based on the following criteria:\n"
        "1. Are the imports in the correct order (e.g., standard library, third-party, local)?\n"
        "2. Are the function names self-explanatory and simple? Suggest better names if needed.\n"
        "3. Are there any redundant variables or functions? Identify and explain them.\n"
        "4. Are the functions logically ordered? Suggest a better order if necessary.\n"
        "5. Are the doc-strings concise and precise?\n"
        "6. Do not modify the code.\n"
        "Provide your review in a clear, easy-to-read format."
    )
    user_query = f"Review the following Python code:\n\n```python\n{code_to_review}\n```"

    try:
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash-preview-05-20',
            system_instruction=system_prompt
        )
        
        print("Attempting to call the Gemini API...")
        response = model.generate_content(user_query)
        
        # Check if the response contains content
        if not response or not response.candidates:
            return "Error: No review found."
            
        return response.text

    except Exception as e:
        return f"Error: Failed to get a review. Details: {e}"


def read_file_content(file_path: str) -> str:
    """Reads the content of a file and returns it as a string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return ""
    except Exception as e:
        return f"Error reading file {file_path}: {e}"

def collect_python_codes(path: str) -> list[str]:
    """
    Collects a list of Python file paths from a given path.
    If the path is a file, it returns a list containing that single file.
    If the path is a directory, it walks through all subdirectories and
    collects all files ending with '.py'.

    Args:
        path (str): The file or directory path to search.

    Returns:
        list[str]: A list of absolute paths to Python files.
    """
    python_files = []
    if os.path.isfile(path):
        if path.endswith(".py"):
            python_files.append(path)
    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
    return python_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review Python code using the Gemini API.")
    parser.add_argument("path", help="Path to a Python file or a directory containing Python files.")
    
    args = parser.parse_args()
    path_to_review = args.path

    if not os.path.exists(path_to_review):
        print(f"Error: The path '{path_to_review}' does not exist.")
    else:
        python_files = collect_python_codes(path_to_review)
        if not python_files:
            print(f"No Python files found at the specified path: {path_to_review}")
        else:
            try:
                # Ensure the output directory exists
                output_dir = 'code_reviews'
                os.makedirs(output_dir, exist_ok=True)

                for file_path in python_files:
                    # Create a unique output filename for each reviewed file
                    file_name_without_ext = os.path.splitext(os.path.basename(file_path))[0]
                    output_file_name = f"review_{file_name_without_ext}.txt"
                    output_path = os.path.join(output_dir, output_file_name)

                    header = f"\n--- Code Review for {file_path} ---\n"
                    print(header)  # Print to console for user feedback

                    code = read_file_content(file_path)
                    if "Error" in code:
                        print(code)
                    else:
                        review_result = review_code(code)
                        print(review_result)  # Print to console for user feedback
                        
                        with open(output_path, 'w', encoding='utf-8') as output_file:
                            output_file.write(header)
                            output_file.write(review_result + "\n\n")

                print(f"\nAll code reviews have been written to the '{output_dir}' directory.")
            except Exception as e:
                print(f"Error: Failed to write to output file. Details: {e}")

