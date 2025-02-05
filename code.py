import os

def read_file_content(file_path):
    """Read and return the content of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading {file_path}: {str(e)}\n"

def main():
    # Directory containing the Python files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of Python files to process (in order)
    files_to_process = [
        'analytics/metrics.py',
        'app.py',
        'config.py',
        'plant.py',
        'train.py',
        'worm_agent.py'
    ]
    
    # Create the output content
    output_content = [
        "# Combined Code from RL-Worm Project\n",
        "This file contains all source code from the RL-Worm project.\n\n"
    ]
    
    # Process each file
    for file_path in files_to_process:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            # Add file header in markdown
            output_content.append(f"# {file_path}\n{'=' * (len(file_path) + 2)}\n\n")
            # Add file content
            output_content.append("```python\n")
            output_content.append(read_file_content(full_path))
            output_content.append("\n```\n\n")
    
    # Write the combined content to a text file
    output_file = os.path.join(base_dir, 'combined_code.txt')
    try:
        with open(output_file, 'w') as f:
            f.writelines(output_content)
        print(f"Successfully created {output_file}")
    except Exception as e:
        print(f"Error writing output file: {str(e)}")

if __name__ == "__main__":
    main()