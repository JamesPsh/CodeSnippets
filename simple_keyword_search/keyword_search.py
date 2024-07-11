import os

def keyword_search(filenames, keywords):
    results = []
    
    for filename in filenames:
        if not os.path.isfile(filename):
            print(f"File {filename} does not exist.")
            continue
        
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        for i, line in enumerate(lines):
            for keyword in keywords:
                if keyword in line:
                    results.append({
                        'filename': filename,
                        'line_number': i + 1,
                        'line': line.strip()
                    })
    
    return results

if __name__ == '__main__':
    # Example usage
    files = ['file1.py', 'file2.py']
    keywords = ['yellow', 'signal']

    search_results = keyword_search(files, keywords)

    # Print search results
    for result in search_results:
        print(f"File: {result['filename']}, Line: {result['line_number']}")
        print(f"Content: {result['line']}")
        print('---')
