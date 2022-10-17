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
Naive Bayes Classifier:
  BAYESIAN CLASSIFICATION REFRESHER: suppose you have a set  of classes
> (e.g. categories) C := {C_1, ..., C_n}, and a  document D consisting
> of words D := {W_1, ..., W_k}.  We wish to ascertain the probability
> that the document  belongs to some class C_j given some set of
> training data  associating documents and classes.
> 
>  By Bayes' Theorem, we have that
> 
>     P(C_j|D) = P(D|C_j)*P(C_j)/P(D).
> 
>  The LHS is the probability that the document belongs to class  C_j
> given the document itself (by which is meant, in practice,  the word
> frequencies occurring in this document), and our program  will
> calculate this probability for each j and spit out the  most likely
> class for this document.
> 
>  P(C_j) is referred to as the "prior" probability, or the  probability
> that a document belongs to C_j in general, without  seeing the
> document first. P(D|C_j) is the probability of seeing  such a
> document, given that it belongs to C_j. Here, by assuming  that words
> appear independently in documents (this being the   "naive"
> assumption), we can estimate
> 
>     P(D|C_j) ~= P(W_1|C_j)*...*P(W_k|C_j)
> 
>  where P(W_i|C_j) is the probability of seeing the given word  in a
> document of the given class. Finally, P(D) can be seen as   merely a
> scaling factor and is not strictly relevant to  classificiation,
> unless you want to normalize the resulting  scores and actually see
> probabilities. In this case, note that
> 
>     P(D) = SUM_j(P(D|C_j)*P(C_j))
> 
![Recommendation App](https://github.com/HungVoCs47/Movie-Recommended-System/blob/master/image/Screenshot%20(1442).png)
