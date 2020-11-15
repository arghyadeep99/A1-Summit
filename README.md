# 1-stop-shortener

1 Stop Shortener tool can be used to summarise:

1. Hindu News Article
2. Wikipedia Articles
3. Simple Text Paragraphs
4. Youtube Videos

It uses an ​extractive approach​ and is ​an unsupervised graph​ based textual content summarization method. It is based on the Page-Rank technique used by search engines like google to provide the top prioritized pages or links to the end -user based on his/her search using this ranking algorithm to calculate the rank of the web-pages.


Watch this demo: https://youtu.be/x9w27wqZdO4

App live at: https://onestopshortener.herokuapp.com/

## Installation

Install the requirements from requirements.txt
```bash
pip install -r requirements.txt
```



## Usage
### 1. Hindu News Article

Enter the Hindu News Article Link in the respective input field and number of sentences required of summarized text in respective input field.

Example
```
Link: https://www.thehindu.com/news/national/pm-modi-congratulates-new-zealand-counterpart-jacinda-ardern-on-poll-win/article32885939.ece
Number of Sentences: 10
```

### 2. Wikipedia Articles

Enter the Wikipedia Article Link in the respective input field and number of words of summarized text in respective input field.

Example
```
Link: https://en.wikipedia.org/wiki/Python_(programming_language)
Number of Words : 300
```

### 3. Simple Text Paragraphs

Enter the paragraphs of text to summarize in the respective text area field and number of sentences required of summarized text in respective input field.

Example
```
Text Area: Copy SampleData.txt into that text area
Number of Sentences: 10
```

### 4. Youtube Videos
Enter the Video id of Youtube video to summarize in the respective input field and number of sentences required of summarized text in respective input field.

Example
```
For Video https://www.youtube.com/watch?v=KR0g-1hnQPA
Enter the following details
Youtube Video Id: KR0g-1hnQPA
Number of Sentences: 5
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


