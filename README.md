<h1 align="center">:books::newspaper: A1-Summit :movie_camera::bookmark_tabs:</h1>

<div align="center">

<img src="./circle-cropped-1.png"></img>

<br>

[![](https://img.shields.io/badge/Made_with-Python3-red?style=for-the-badge&logo=python)]("Python3")
[![](https://img.shields.io/badge/Framework-Streamlit-red?style=for-the-badge&logo=streamlit)]("Streamlit")
[![](https://img.shields.io/badge/Deployed_on-Heroku-red?style=for-the-badge&logo=heroku)](https://www.heroku.com/  "Heroku")

<br>

</div>

## What is A-1 Summit?

A-1 Summit is an "All-in-1" Summariser, where you can summarise:

1. The Hindu News Article
2. Wikipedia Articles
3. Simple Text Paragraphs
4. YouTube Videos (transcript enabled)

It uses an **extractive approach** and is an **unsupervised graph** based textual content summarization method. It is based on the **Page-Rank Algorithm** used by search engines like Google to provide the top prioritized pages or links to the end-user based on his/her search using this ranking algorithm to calculate the rank of the web-pages.

App live at: https://a1-summit.herokuapp.com/

## To try locally in your machine: 

1. Clone this repository: 

 `git clone https://github.com/arghyadeep99/A1-Summit.git`
 
2. Install the requirements from requirements.txt.

`pip install -r requirements.txt`

3. cd into the A1-Summit repository and run:

`streamlit run A1-Summit.py`

## Usage
### 1. Hindu News Article

Enter the Hindu News Article Link in the respective input field and number of sentences required of summarized text in respective input field.

Example
```
Link: https://www.thehindu.com/news/international/trump-appears-to-acknowledge-for-first-time-that-biden-could-succeed-him/article33099094.ece
Number of Sentences: 10
```

### 2. Wikipedia Articles

Enter the Wikipedia Article Link in the respective input field and **number of words** of summarized text in respective input field.

Example
```
Link: https://en.wikipedia.org/wiki/Kharghar
Number of Words : 100
```

### 3. Simple Text Paragraphs

Enter the paragraphs of text to summarize in the respective text area field and number of sentences required of summarized text in respective input field.

Example
```
Text Area: Copy SampleData.txt into that text area
Number of Sentences: 10
```

### 4. YouTube Videos
Enter the Video id of Youtube video to summarize in the respective input field and number of sentences required of summarized text in respective input field.

Example:
For Video https://www.youtube.com/watch?v=KR0g-1hnQPA, enter the following details:
```
Youtube Video Id: KR0g-1hnQPA
Number of Sentences: 10
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


