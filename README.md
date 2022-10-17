# Movie-Recommended-System
![Python](https://img.shields.io/badge/Python-3.8-blueviolet)
![Framework](https://img.shields.io/badge/Framework-Flask-red)
![Frontend](https://img.shields.io/badge/Frontend-HTML/CSS/JS-green)
![API](https://img.shields.io/badge/API-TMDB-fcba03)
## Overview
Recommender system has the ability to predict whether a particular user would prefer an item or not based on the user's profile. Recommender systems are beneficial to both service providers and users. The movies are recommended based on the content of the movie the users entered or selected. This will bring back more opportunities for customer to choose the films which is relatively similar types(horror, adventure, sci-fi, ...) to their favorite types of movie. Furthermore, the recommended system also employ Sentiment Analysis function in order to figure out the feeling of the users about the film (positive/negative).
## DEMO
![Recommendation App](https://github.com/HungVoCs47/Movie-Recommended-System/blob/master/image/Screenshot%20(1440).png)














![Recommendation App](https://github.com/HungVoCs47/Movie-Recommended-System/blob/master/image/Screenshot%20(1439).png)

## METHOD
For the recommendeded system, cosine similarty is applied to search the relative movies in the list of all movies. Weights are assigned to movies, the more similar of the movie, the higher weight is assigned to the movie\
For the Sentiment Classifier, the system is used simple supervised algorithm Naive Bayes Classifier. However, this algorithm is "naive" because the structure (grammar) of the sentence is neglected and the words are independent to other. However, it is still one of the effective method for analyzing the sentiment of a sentence.\
![Recommendation App](https://github.com/HungVoCs47/Movie-Recommended-System/blob/master/image/Screenshot%20(1442).png)
