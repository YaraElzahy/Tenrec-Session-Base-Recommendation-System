# Tenrec-Session-Base-Recommendation-System (Recomind)

## A Session Based - Recommendation System

This System is a session-based recommendation system, powered by the Ternec dataset. Leveraging the [Tenrec: A Large-scale Multipurpose Benchmark Dataset for Recommender Systems](https://proceedings.neurips.cc/paper_files/paper/2022/file/4ad4fc1528374422dd7a69dea9e72948-Paper-Datasets_and_Benchmarks.pdf) Paper, we aim to deliver personalized recommendations that adapt to users' session-specific preferences in real-time. Join us as we revolutionize the way users discover content by harnessing the power of session-based modeling and the rich insights provided by the Ternec dataset.

## Team Members
- Adham Mokhtar
- Manar El-Ghobashy
- Yara Hassan
- Yara El-Zahy

## Some Explainations

A recommendation system is like having a smart computer friend that suggests things you might like based on the things you already enjoy. It helps you discover new things that you'll love!

There are three main types of recommendation systems:
1. Content-Based: It recommends things based on the features of what you already like. For example, if you enjoy action movies, it will suggest more action movies.
2. Collaborative Filtering: It recommends things based on what other people with similar tastes enjoy. If someone similar to you likes a certain video game, it will suggest that game to you.
3. Hybrid: This combines different methods to give you even better recommendations. It uses both the features of what you like and what other people like to suggest things you'll enjoy.

"Session-based" and "cold start" are specific challenges or scenarios that can occur in recommendation systems:
- Session-based Recommendations: This means giving recommendations based on what you're doing right now. For example, if you're browsing a website, it suggests things related to what you're looking at.
- Cold Start Problem: This happens when the system doesn't know much about a new user or item. It's like when you join a new website or they add new things. The system has to find other ways to give you recommendations since it doesn't have much information yet.

For session-based recommendations, a type of recommendation system called "Sequential Recommendation" is more suitable.

- Sequential Recommendation systems are specifically designed to make recommendations based on a user's current session or sequence of actions within a single browsing session. These systems analyze the order and context of the user's interactions to understand their immediate preferences and provide relevant suggestions.
- In a session-based recommendation system, the focus is on capturing the user's current interests and recommending items that align with those interests. This type of system takes into account the sequence of actions, such as clicks, views, or purchases, during a session to make accurate and timely recommendations.

In this project, We experiment different models namely NextItNet, GRU4rec, BERT4rec and SASrec.

## Recommended Pipline

- Data Preprocessing
- Session Representation
- Train-Test-Validation Split
- Model Selection
    - Model Training
    - Model Evaluation
    - Hyperparameter Tuning
    - Performance Analysis

## Data Acquisition

The four datasets used in this paper are raw datasets:
- `QK-video.csv`
- `QK-article.csv`
- `QB-video.csv`
- `QB-article.csv`

All task-specific datasets are extracted from these datasets.

Subdataset:

1. `ctr_data_1M.csv` is used in CTR task (Section 3.1) and Multi-Task Learning  (Section 3.3).

2. `cold_data.csv` (Table 7, Section 3.6), `cold_data_1.csv`, `cold_data_0.3.csv`, `cold_data_0.7.csv` are used for the Cold-Start task (see Appendix Table 5).

3. `task_0.csv`, `task_1.csv`, `task_2.csv`, `task_3.csv` are used in Lifelong Learning (see Section 3.7, Table 8).

4. `sbr_data_1M.csv` is used in Session-based Recommendation (Section 3.2), Transfer Learning (Section 3.4, used as pre-training dataset), User Profile Prediction (Section 3.5), Model Compression (Section 3.8), Model Training Speedup (Section 3.9).

Note that:

1. Model Inference Speedup Task (Section 3.10): the dataset is `QB-video.csv`, and Transfer Learning Task (Section 3.4): target dataset is also `QB-video.csv`.

2. We sort the items at the user level in order of click time, so the time information is implicit in the order of the items.

Example:
| userid | itemid |
| ---    | ---    |
| 2345   | 12     |
| 2345   | 5      |
| 2345   | 61     |
| 2345   | 78     |
| 2345   | 35     |

The click sequence of user 2345 is [12, 5, 61, 78, 35].

We report baseline results evaluated on QK-video-1M here. Results of the full QK-video
datasets will be present in the leaderboard. Following the common practice [61], we simply filter
out sessions with length shorter than 10. Given that the average session length is 28.34, we set the
maximum session lengths to 30. Session length less than 30 will be padded with zero, otherwise only
recent 30 interactions are kept. After pre-processing, we obtain 928,562 users, 1,189,341 items and
37,823,609 clicking interactions. We keep the last item in the session for testing, the second to last
for validating, and the remaining for training.

So we will use `sbr_data_1M.csv` in this project.



## Collaborative Filtering

Ways to determine whether user-based or item-based collaborative filtering is more suitable:

- User-based filtering is better when you have a large number of users with consistent preferences and behaviors, and when users have interacted with a significant portion of the available items.
- Item-based filtering is better when you have a large number of items with consistent characteristics and interactions, and when users have interacted with only a small fraction of the available items.

Since the number of items in the dataset is larger than the number of users, an Item-Based filtering approach appears to be more suitable.


## Data Selection

- We are preparing a subset of the data for testing the generalization of the session-based recommendation model.
- We select users with age values of 0 and 3, as they have similar record counts, user counts, and item counts to be away of data bias.
- Due to GPU resource limitations and previous training failures on Colab and the university server, we are restricting the dataset to one million records
- This filtered data will be used to evaluate the model's performance and assess if it can maintain a high value of NDCG@20 (Normalized Discounted Cumulative Gain at top-20) on this combination of age groups.

## Session Representation

- Items Label Encoding
- Session Initialization
- Session Length Filtering
- Session Sorting - already sorted
- Session Padding


## Results

we explore and analyze the performance of our best session-based recommendation model. We selected the NextItNet model as it outperformed other candidate models, including GRU4Rec, BERT4Rec, and SASRec, based on the NDCG@20 evaluation metric. Our comparison was conducted on two distinct datasets: one containing users with age 3 and the other with users having age 0.

The evaluation results for each model on both user groups are as follows:
| Model | NDCG@20 for users with age 3 | NDCG@20 for users with age 0 |
| --- | --- | ---|
| NextItNet| 0.0112 | 0.0086 |
| SASRec | 0.0103 | 0.0079 |
| GRU4Rec | 0.0100 | 0.0058 |
| BERTModel | 7.7849e-05 | 1.1388e-05 |

From the comparison, it is evident that the NextItNet model consistently outperformed the other models on both user groups, showcasing its ability to capture sequential patterns effectively for session-based recommendation tasks.











