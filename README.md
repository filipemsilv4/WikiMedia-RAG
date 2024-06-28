# RAG_test EN:
RAG using Google Text Embeddings and Gemini Flash

## How to use:
- Get a [Gemini API key](https://ai.google.dev/gemini-api/docs/api-key)
  
- If you don't have one yet, create a virtual environment in a directory of your choice  
  ```python -m venv <directory>```
  
- Activate the virtual environment
    - MacOS/Linux: ```source <directory>/bin/activate```  
    - Windows (PowerShell): ```<directory>\Scripts\Activate.ps1```  
    - Windows (CMD): ```<directory>\Scripts\activate.bat```
      
- Set an environment variable called "GOOGLE_API_KEY" with your Gemini API key  
    ```export GOOGLE_API_KEY=your_API_key```
  
- Install the project dependencies
    ```pip install -r requirements.txt```
  
- Get an [XML export of the MediaWiki pages you want to use as source](https://www.mediawiki.org/wiki/Manual:Importing_XML_dumps)
    - Dump files can be obtained on the Special:Export page of the Wiki
      
- Place the XML file in ```data/comics.xml```
  
- Run the program with ```python rag.py```.
