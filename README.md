# SemEval-2017-Task-5
![alt text](img.png)

This repository contains the source codes produced by **RiTUAL-UH** team for participating in **SemEval-2017-Task-5: Fine-Grained Sentiment Analysis on Financial Microblogs and News**. 
> - **RiTUAL-UH** ranked **2nd** in subtask-2 (News Headlines) and **6th** in subtask-1 (Financial Microblogs). 
> - Using alternative metrics that incorporates company informations, **RiTUAL-UH** ranked **1st** in both of the subtasks.

#### For more Details
[More Details of the task could be found in the Overview Paper.](http://nlp.arizona.edu/SemEval-2017/pdf/SemEval089.pdf)

[More Details about the system could be found in the System Description Paper.](http://sudiptakar.info/wp-content/uploads/2018/02/semeval2017task5.pdf)

#### Bibtex to cite this paper
```
@InProceedings{kar-maharjan-solorio:2017:SemEval,
  author    = {Kar, Sudipta  and  Maharjan, Suraj  and  Solorio, Thamar},
  title     = {RiTUAL-UH at SemEval-2017 Task 5: Sentiment Analysis on Financial Data Using Neural Networks},
  booktitle = {Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)},
  month     = {August},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {877--882},
  abstract  = {In this paper, we present our systems for the “SemEval-2017 Task-5 on Fine-
	Grained Sentiment Analysis on Financial Microblogs and News”. In our system,
	we combined hand-engineered lexical, sentiment and metadata features, the
	representations learned from Convolutional Neural Networks (CNN) and
	Bidirectional Gated Recurrent Unit (Bi-GRU) with Attention model applied on
	top. With this architecture we obtained weighted cosine similarity scores of
	0.72 and 0.74 for subtask-1 and subtask-2, respectively. Using the official
	scoring system, our system ranked the second place for subtask-2 and eighth
	place for the subtask-1. It ranked first for both of the subtasks by the scores
	achieved by an alternate scoring system.},
  url       = {http://www.aclweb.org/anthology/S17-2150}
}

```

#### Directory Description
- source_project : Pycharm project containing source codes written in Python

-- [experiments] : codes of the models

-- [prepare_data] : codes for creating sequences and other preprocessing operations

-- [features] : lexical and embedding feature extraction functions
- submission : Predicted Sentiment scores for the test data 

#### Contact
Sudipta Kar 
email: skar3 AT uh DOT edu

