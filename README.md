# Mental Health as a Function of Music - Data Mining 

An exploratory research to find any correlation between a listener's preference in music and their mental health.

Python and Spotipy (Python Spotify Web Wrapper) were used to create the data mining models and to retrieve song information from Spotify's Web API.


## Data Collection

The primary data was collected through a survey that asked surveyers to list their 3 favourite songs at the moment and to rate their mental health using a likert scale.

- ability to enjoy life
- resilience
- balanced lifestyle
- emotional flexibility
- self-actualization

The survey also collected general data such as:

- gender
- age range
- hours of music they listen to per day
- if they have experienced anything unusual/traumatic recently (to help ensure music is the main factor contributing to mental health)

Once the survey had more than 300 entries (and approximately 1000 songs), a Python script was made to fetch data from Spotify's music catalog using Spotify Web API. The  [Get Audio Features for a Track](https://developer.spotify.com/web-api/get-audio-features/) endpoint was used to retrieve song information in a JSON format. An example of this JSON output is shown below for a song.

<b>"Starboy - The Weeknd ft. Daft Punk"</b>

[{
- "track_href": "https://api.spotify.com/v1/tracks/7MXVkk9YMctZqd1Srtv4MB",
- "type": "audio_features",
- "analysis_url": "https://api.spotify.com/v1/audio-analysis/7MXVkk9YMctZqd1Srtv4MB",
- "acousticness": 0.168,
- "speechiness": 0.284,
- "liveness": 0.136,
- "uri": "spotify:track:7MXVkk9YMctZqd1Srtv4MB",
- "id": "7MXVkk9YMctZqd1Srtv4MB",
- "key": 7,
- "mode": 1,
- "time_signature": 4,
- "duration_ms": 230453,
- "tempo": 185.998,
- "energy": 0.595,
- "danceability": 0.675,
- "valence": 0.49,
- "instrumentalness": 3.36e-06
}]

For the purposes of this study, only the following features were retrieved: 

| Feature  | Definition by Spotify  |
|:-:|---|
| Energy |Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. |
| Danceability | Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.  |
| Valence  |  A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).  |
| Tempo  | The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. |
| Popularity  | The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are.  |
| Instrumentalness  | 	Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.  |
|  Acousticness | 	A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.  |
| Liveness  | Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live. |

Once the data was collected, it was cleaned to ensure that there were no missing fields. Furthermore, each listener's total mental health score was calculated as a sum of the individual mental health answers and an average of their musical properties were taken to reflect the listener's overall musical behaviour. 

Now that all the data has been collected and cleaned, time to get into the more interesting material!

## Exploratory Data Analysis

Let's begin by plotting bar charts and histograms of each variable obtained from the survey and spotify to understand it's distribution, mode, variance etc. 

53.8% of the data set consists of females, 45.3% males and less than 1% chose not to specify.

Surprisingly, <b>46.6%</b> of the data set listens to 2+ hours of music, <b>35%</b> listen to 1-2 hours and <b>18.4%</b> listen to 0 - 1 hours of music. It was expected that most people would be listening to 1-2 hours, but we underestimated the number of avid music listeners. 

Similarly, our data set primariy consisted of 18-30 year olds. We realzied later that this was too large of a range and should have been divided further.


Tempo, popularity, energy, dance and valence have a nice normal distribution which shows that listeners listen to a variety of music hovering around a mean of <b> 123 bpm, 61 popularity, 0.65 energy, 0.60 danceability, 0.45 valence</b>. The entire dataset is primarily not listening to instrumental tracks as shown by the instrumentalness graph which means we can drop this feature as it is least likely to be contributing to mental health. However let's keep exploring the data set before removing any features.


The mental health histogram is also normally distributed around a mental health
score between 21-23. There are significantly less people with a low mental health
score as majority of the people have a medium to high mental health score.



The scatter plot does not show any clear relationship between mental health and any musical feature (such as linear, logarithmic etc). However, we are able to conclude that instrumentalness is a weak feature in potentially predicting mental health as mental health ranges from 5 to 30 for really low instrumental values. To say this conclusively, let's do some feature engineering to ensure the right attributes are being used to predict mental health.

## Feature Engineering

### Recursive Feature Elimination (RFE)

The Recursive Feature Elimination (RFE) method is a feature selection approach. It works by recursively removing attributes and building a model on those attributes that remain. It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute. Using sklearn's LogisticRegression and RFE library, the following features were found to have the most impact on the mental health class variable. 

| Rank  | Song Attribute  |
|:-:|---|
|  1 | Energy  |
|  2 | Danceability |
|  3 | Popularity  |
|  4 | Tempo  |
|  5 | Valence  |
|  6 | Acousticness  |
|  7 | Liveness  |
|  8 | Instrumentalness  |

As expected, instrumentalness is the least important feature along with livness and acousticness. Therefore, these three features were removed for the remainder of the study in building more models.

## Unsupervised Learning

### Association 

The purpose of association was to find patterns in the data along with analyzing the relationships between various attributes. Based on the scatter plots analyzed previously, it was hypothesized that with a high confidence level, the categorical variable mental health will not be part of any rules. This is because mental health did not consistently correlate with any of the song attributes. In addition, 87.5% of the dataset consisted of people from ages 18-20 years old. It is assumed that the age range categorical variable will be associated with a lot of attributes due to it abundance of 18-20 year olds in the dataset.

The APRIORI algorithm was used for association rule mining. Since there were only 255 unique records, a minimum support of ten percent was chosen so each rule would cover at least 25 records. If there was a much larger dataset, a lower support percentage would have been chosen. For example, for a million records, a support of one percent would be chosen as it covers 10,000 records. A support of 10-20 percent is a reasonable assumption for the current dataset. 

As expected, association rules for mental health had a very low confidence. There are several reasons as to why the APRIORI algorithm did not generate good association rules for mental health based on the dataset. One factor could be that music may not be a dominant influence on a person’s mental health leading to poor association rules. Another factor could be the lack of song data. However, some interesting associations amongst features are shown below: 

| Rule | Association | Confidence |
|:-:|---|---|
| High Popularity + High Valence | High Danceability  | 96.7% |
| Low Popularity + Low Valence | Low Danceability  | 94.1% |
| High Energy + Fast Tempo + No Traumatic Experience | Low Danceability | 75% |
| High Danceability + Medium Popularity + High Valence | Female | 70% |
| High Dance + Low Energy  | No Traumatic Experience  | 70% |

These were a few of the sensible rules out of thousands of rules. "High Popularity + High Valence -> High Danceability" and vice versa could indicate that songs that perhaps songs that get popular on Spotify may primarily be highly danceable songs. The other rules are interesting as well, such as "High Danceability + Medium Popularity + High Valence" has a 70% confidence of being a female. Let's see what type of clusters we can identify to see what "type" of people exist in the dataset. 


| Field  | Value  |
|:-:|---|
| Coefficients (m))  | 15.21 |
| Intercept (b) | 8.94 |
| Equation of the Line  | Mental Health = (15.21)(danceability) + 8.94 |
| Mean Residual Sum of Squares  | 19.59  |
| RMSE  |  4.42  |
| R-Squared  | 0.14  |
| Standard Deviation of Results |  4.42  |

The residual plot of mental health scores against danceability showed a normal distribution concentrated around zero. This validates that a linear regression model is an appropriate model although not an applicable one due to such a wide spread of data points. The R-squared value obtained for the linear regression model was 14% which is fairly low. This further shows that the model does not account for much variability in
the data.


The R-squared value increases from 13% in the linear regression to 20.5% in the multi linear regression. However, it is still a weak r-squared value. However, the adjusted r-squared is significantly lower at 15.8%. The adjusted R-squared is a modified version of R-squared that has been adjusted for the number of predictors in the model. This could mean that even 5 features are too many (or too few) to predict mental health and that the gap between r-squared and adjusted r-squared could be less by adding or removing features. 

Regardless, it is conclusive that a linear model is not an accurate model to represent mental health and musical features. Let's keep searching!


### Naive Bayes Model

The last model that was considered to make predictions was Naive Bayes model. Naive Bayes is a classification technique based on Bayes’ Theorem with an assumption of independence amongst its features/predictors.Bayes’ Theorem can be used to calculate the probability of a person’s mental health category given the numerical song attributes of their preferred songs. It is important to note the assumption that each song attribute is independent of another for Naïve Bayes’ theorem.

The top 4 features were selected as they had the highest accuracy of the model 0.43. The model has a really weak prediction accuracy, with the following confusion matrix: 

|   |   |   | Predicted  |   |
|:-:|---|---|---|---|
|   |   | High | Medium |  Low |
|   | High  | 37 | 45 | 13 |
| <b>Actual</b> | Medium | 19 | 62 | 11 |
|   | Low | 33 | 24 | 11 |

The surprising weakness was that the model incorrectly predicted 33 data points to have a high mental health when they have a low mental health. This is the most glaring problem with the model. Since medium health falls in the middle of high and low health, some overlap in prediction of medium mental health is expected. However, incorrectly predicting high mental health when it is in fact low mental health, or vice versa is a significant flaw. This is due to the scattered distribution of the data points.

To ensure that the model is not overfitting, the data was cross validated with 10 folds. The mean of the 10 fold cross-validation was calculated to be 0.38. Therefore, the model is slightly overfitting. However, it is within 10% so the model is not extremely overfitting. A potential solution to reduce overfitting would be to gather more data from people who have low mental health since the dataset has few people with low mental health, as seen in the histogram. The Naive Bayes’ model was used to predict mental health using the 4 features; energy, danceability, popularity and tempo. These predictions are shown below: 

|  Description | Energy  | Dance  | Popularity   | Tempo  | Result Health   | Confidence   |
|:-:|---|---|---|---|---|---|
| High Value Features | 0.95 | 0.95 | 100 | 150 | High | 70.3% |
| Mid-High Value Features | 0.7 | 0.7 | 70 | 140 | High | 40.2% |
| Medium Value Features | 0.6 | 0.6 | 60 | 120 | Medium | 47.1% |
| Low Value Features | 0.2 | 0.2 | 20 | 100 | Low | 93.3% |

Although the accuracy of the model is weak, the model of the output is as hypothesized: users who listen to songs with high dance, energy, tempo and popularity are 63% probable to have a high mental health. Conversely, those who listen to all features with extremely low values are 93.3% probable to have low mental health. Therefore the model is useful to predict mentalhealth when the feature values are extreme, either extremely high or low. However, the Naive Bayes’ model struggles to predict mental health when the song attributes hover around 0.5, as shown by the probability of prediction for each cluster. The model struggles to predict mid-ranged values as most values in the dataset are centered around the middle, as seen by the scatterplot figures.

## Recommendations

Naive Bayes’ can be used as a powerful tool to recommend songs to users to improve their mental health since it predicts low and high mental health with a strong probability for extreme values of each song attributes. Therefore, if it is known that a person has low mental health, songs can be recommended to users that have a high probability of improving mental health due to the Naive Bayes’ prediction. In addition to individual songs, a playlist which averages out to yield a high mental health prediction could be recommended.

The combination of the above songs can improve your stress level and mental health according to our model, as these songs combined are predicted to have a high mental health with a probability of 60%”.

## Conclusion

Data mining procedures such as linear regression, association, clustering and Naïve Bayes’ were used to analyze the effect that music has on a person’s overall mental health. The song attributes of each song inputted by individuals were obtained by using Spotify's Web API.

Based on the results of association rule mining, zero song attributes were associated with any of the categorical mental health variables if a high confidence level was used. In addition, the best linear regression model (mental health vs danceability) has an R-squared value of 13%
which does not account for much variability in the dataset. Naïve Bayes’ model has a low accuracy but provides usable predictions and matches the initial hypothesis. However, there are several reasons as to why the results of the data mining procedures were not favorable. This could be due to the lack of song data as only three songs were collected per individual. Three songs are not an accurate representation of an individual’s music preferences.

